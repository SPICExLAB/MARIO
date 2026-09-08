import os
import sys
import torch
import pypose as pp

import torch.utils.data as Data
import argparse
import pickle

import tqdm
from utils.utils import move_to, save_state
from pyhocon import ConfigFactory

from datasets.dataset_utils import collate_fcs
from datasets.dataset_motion import SeqeuncesMotionDataset
from model import net_dict
from utils import *

def inference(network, loader, confs, pose_predictor=None, step_size=None, window_size=None):
    '''
    Correction inference with support for overlapping windows.
    
    When step_size < window_size, windows overlap. For overlapping regions:
    - Only keep the last `step_size` frames from each window (except the first window)
    - This simulates online inference where we use the most recent prediction
    
    Args:
        network: The neural network model
        loader: DataLoader with windowed data
        confs: Configuration object
        pose_predictor: Optional pose predictor model
        step_size: Step size for sliding window (for handling overlaps)
        window_size: Window size (for handling overlaps)
    '''
    network.eval()
    evaluate_states = {}
    use_raw_quat = getattr(confs, 'use_raw_quat', False)
    
    # Determine if we have overlapping windows
    has_overlap = (step_size is not None and window_size is not None and step_size < window_size)
    
    # We need to compute the output step size based on network downsampling
    # This will be computed after first forward pass when we know actual output size
    output_step_size = None
    
    if has_overlap:
        print(f"[INFO] Processing overlapping windows with input step_size={step_size}")
    
    with torch.no_grad():
        inte_state = None
        for i, (data, _, label) in enumerate(tqdm.tqdm(loader)):
            # Move data to device
            data, label = move_to([data, label], confs.device)
          
            for k, v in data.items():
                if isinstance(v, torch.Tensor):
                    data[k] = v
                    
            # Extract rotation
            try:
                if use_raw_quat:
                    # Use quaternions directly
                    if hasattr(label['gt_rot'], 'quaternion'):
                        rot = label['gt_rot'][:,:-1,:].quaternion()  # [B, T-1, 4]
                    else:
                        # Already quaternion tensor
                        rot = label['gt_rot'][:,:-1,:]  # Assume [B, T, 4]
                    if hasattr(label['gt_rot'], 'Log'):
                        rot_ = label['gt_rot'][:,:-1,:].Log().tensor()
                    else:
                        # If it's already a regular tensor, assume it's in quaternion format
                        rot_ = pp.SO3(label['gt_rot'][:,:-1,:]).Log().tensor()
                else:
                    # Original pipeline: convert to log
                    if hasattr(label['gt_rot'], 'Log'):
                        rot = label['gt_rot'][:,:-1,:].Log().tensor()
                    else:
                        # If it's already a regular tensor, assume it's in quaternion format
                        rot = pp.SO3(label['gt_rot'][:,:-1,:]).Log().tensor()
            except Exception as e:
                print(f"[ERROR] Failed to process rotation: {e}")
                raise
            
            data["gt_rot"] = pp.SO3(label['gt_rot'])
        
            # Forward pass
            try:
                if pose_predictor is not None:
                    T = data['acc'].shape[1]
                    # with torch.no_grad():
                    #     acc_50hz = data['acc'][:, ::4, :]  # Downsample to 50Hz
                    #     gyro_50hz = data['gyro'][:, ::4, :]  # Downsample to 50Hz
                    #     pose_data = {'acc': acc_50hz, 'gyro': gyro_50hz}
                    #     if rot.shape[-1] == 4:
                    #         pose_rot = rot_[:, ::4, :] if rot_.dim() == 3 else None
                    #     else:
                    #         pose_rot = rot[:, ::4, :] if rot.dim() == 3 else None
                    #     joint_pred = pose_predictor(pose_data, pose_rot)
                    #     joint_rot = joint_pred['joint_rot']
                    #     joint_rot = joint_rot.repeat_interleave(4, dim=1)
                    #     joint_rot = joint_rot[:, :T, :]
                    # inte_state = network(data, rot, joint_rot=joint_rot) # 200Hz
                    with torch.no_grad():
                        if confs.get('resample_hz', None) is not None:
                            resample_factor = int(confs.resample_hz/50)
                        else:
                            resample_factor = 4  # Default: 200Hz -> 50Hz
                        acc_resampled = data['acc'][:, ::resample_factor, :]  # Downsample to target Hz
                        gyro_resampled = data['gyro'][:, ::resample_factor, :]  # Downsample to target Hz
                        pose_data = {'acc': acc_resampled, 'gyro': gyro_resampled}
                        # When use_raw_quat=True, rot_ contains log-space rotations for pose predictor
                        # When use_raw_quat=False, rot is already in log space
                        if use_raw_quat:
                            pose_rot = rot_[:, ::resample_factor, :] if rot_.dim() == 3 else None
                        else:
                            pose_rot = rot[:, ::resample_factor, :] if rot.dim() == 3 else None
                        joint_pred = pose_predictor(pose_data, pose_rot)
                        joint_rot = joint_pred['joint_rot']
                        joint_rot = joint_rot.repeat_interleave(resample_factor, dim=1)
                        joint_rot = joint_rot[:, :T, :]
                    inte_state = network(data, rot, joint_rot=joint_rot) # 200Hz
                else:
                    inte_state = network(data, rot)
            except Exception as e:
                print(f"[ERROR] Network forward pass failed: {e}")
                raise

            # Get timestamp and save state
            # Handle DataParallel wrapper
            if hasattr(network, 'module'):
                inte_state['ts'] = network.module.get_label(data['ts'][...,None])[0]
            else:
                inte_state['ts'] = network.get_label(data['ts'][...,None])[0]
            
            # For overlapping windows, only keep the relevant portion
            if has_overlap and i > 0:
                # Compute output_step_size on first window with overlap handling
                if output_step_size is None:
                    # Get actual output size from network
                    output_size = inte_state['net_vel'].shape[-2]  # e.g., 112 for 1000 input
                    # Compute downsampling factor
                    downsample_factor = window_size / output_size  # e.g., 1000/112 ≈ 8.93
                    # Scale step_size to output space (round up to ensure we don't miss frames)
                    output_step_size = max(1, int(round(step_size / downsample_factor)))
                    print(f"[INFO] Network downsamples {window_size} -> {output_size} (factor={downsample_factor:.2f})")
                    print(f"[INFO] Input step_size={step_size} -> output_step_size={output_step_size}")
                
                # For windows after the first, only keep the last `output_step_size` frames
                # This corresponds to the non-overlapping portion in output space
                for k, v in inte_state.items():
                    if isinstance(v, torch.Tensor) and v.dim() >= 2:
                        # Assuming time dimension is -2 (second to last)
                        inte_state[k] = v[..., -output_step_size:, :]
            
            save_state(evaluate_states, inte_state)

        # Concatenate results
        for k, v in evaluate_states.items():
            evaluate_states[k] = torch.cat(v, dim=-2)
    return evaluate_states

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/EuRoC/motion_body.conf', help='config file path')
    parser.add_argument('--load', type=str, default=None, help='path for specific model check point, Default is the best model')
    parser.add_argument('--ckpt', type=str, default=None, help='explicit checkpoint path; overrides <exp_dir>/ckpt/best_model.ckpt')
    parser.add_argument("--device", type=str, default="cuda:0", help="cuda or cpu")
    parser.add_argument('--batch_size', type=int, default=1, help='batch size.')
    parser.add_argument('--seqlen', type=int, default=1000, help='window size.')
    parser.add_argument('--stepsize', type=int, default=None, help='step size for sliding window (default: same as seqlen). Use smaller values for online inference.')
    parser.add_argument('--whole', default=False, action="store_true", help='estimate the whole seq')
    parser.add_argument(
        "--multi_gpu", default=False, action="store_true", 
        help="Use all available GPUs with DataParallel"
    )
    parser.add_argument(
        "--gpu_ids", type=str, default=None,
        help="Comma-separated GPU IDs to use (e.g., '0,1,2'). Used with --multi_gpu"
    )


    args = parser.parse_args()
    
    # Handle GPU configuration
    if args.multi_gpu:
        if args.gpu_ids:
            gpu_ids = [int(id.strip()) for id in args.gpu_ids.split(',')]
            args.device = f"cuda:{gpu_ids[0]}"
            print(f"Using multiple GPUs: {gpu_ids}")
        else:
            gpu_ids = list(range(torch.cuda.device_count()))
            args.device = "cuda:0"
            print(f"Using all available GPUs: {gpu_ids}")
    else:
        gpu_ids = None
        print(f"Using device: {args.device}")
    
    print(args)
    conf = ConfigFactory.parse_file(args.config)
    conf.train.device = args.device
    conf_name = os.path.split(args.config)[-1].split(".")[0]
    conf['general']['exp_dir'] = os.path.join(conf.general.exp_dir, conf_name)
    conf['device'] = args.device
    dataset_conf = conf.dataset.inference

    # Special handling for wrapper: load fresh body model to avoid BatchNorm corruption
    if conf.train.network == 'world_velocity_wrapper':
        print("\n🔧 Loading wrapper with fresh body model to prevent BatchNorm corruption...")

        # Load fresh body model from pretrained path
        pretrained_path = getattr(conf.train, 'pretrained_body_model', None)
        if pretrained_path is None:
            raise ValueError("Wrapper config must specify 'pretrained_body_model' path")

        # Get body network type from config
        from model.code import (
            TransformerMotionNormQuatDiff_v0_light,
            CodeNetMotionwithRot,
            CodeNetMotionwithRot_Pose
        )

        BODY_MODEL_CLASSES = {
            'transformernormquatdiff_v0_light': TransformerMotionNormQuatDiff_v0_light,
            'codewithrot': CodeNetMotionwithRot,
            'codewithrot_pose': CodeNetMotionwithRot_Pose,
        }

        body_network_name = getattr(conf.train, 'body_network', 'transformernormquatdiff_v0_light').lower()
        if body_network_name not in BODY_MODEL_CLASSES:
            raise ValueError(f"Unknown body_network: {body_network_name}. "
                           f"Options: {list(BODY_MODEL_CLASSES.keys())}")

        print(f"   Loading fresh body model ({body_network_name}) from: {pretrained_path}")
        body_checkpoint = torch.load(pretrained_path, map_location=args.device, weights_only=True)

        # Create fresh body model instance
        body_model_class = BODY_MODEL_CLASSES[body_network_name]
        body_model = body_model_class(conf.train).to(args.device)

        # Load body model weights
        body_state_dict = body_checkpoint['model_state_dict']
        new_body_state_dict = {}
        for k, v in body_state_dict.items():
            if k.startswith('module.'):
                new_body_state_dict[k[7:]] = v
            else:
                new_body_state_dict[k] = v
        body_model.load_state_dict(new_body_state_dict, strict=True)
        print(f"   ✓ Fresh body model ({body_network_name}) loaded successfully")

        # Create wrapper with fresh body model
        network = net_dict[conf.train.network](conf.train, body_model=body_model).to(args.device)
        print("   ✓ Wrapper created with fresh body model\n")
    else:
        network = net_dict[conf.train.network](conf.train).to(args.device)
    
    if conf.get('prior', None) is not None:
        pose_predictor = net_dict[conf.prior.network](conf.prior).to(
        device=args.device, dtype=torch.float32
    )
        pose_predictor_checkpoint = torch.load(
                conf.prior.pretrained_path,
                map_location=args.device,
                weights_only=True
            )
        pose_predictor.load_state_dict(pose_predictor_checkpoint["model_state_dict"])
        pose_predictor.eval()
    else:
        pose_predictor = None
        
    # Wrap network with DataParallel if using multiple GPUs
    if args.multi_gpu and torch.cuda.device_count() > 1:
        if gpu_ids:
            network = torch.nn.DataParallel(network, device_ids=gpu_ids)
        else:
            network = torch.nn.DataParallel(network)
        print(f"Model wrapped with DataParallel, using {len(network.device_ids)} GPUs")
    
    save_folder = os.path.join(conf.general.exp_dir, "evaluate")
    os.makedirs(save_folder, exist_ok=True)

    if args.ckpt:
        ckpt_path = args.ckpt                      # explicit checkpoint (e.g. a Nymeria model evaluated on Aria)
    elif args.load is None:
        ckpt_path = os.path.join(conf.general.exp_dir, "ckpt/best_model.ckpt")
    else:
        ckpt_path = os.path.join(conf.general.exp_dir, "ckpt", args.load)

    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=torch.device(args.device),weights_only=True)
        print("loaded state dict %s in epoch %i"%(ckpt_path, checkpoint["epoch"]))
        
        # Handle DataParallel checkpoint loading
        state_dict = checkpoint["model_state_dict"]
        if args.multi_gpu and torch.cuda.device_count() > 1:
            # Check if the checkpoint was saved with DataParallel
            if not any(k.startswith('module.') for k in state_dict.keys()):
                # Add 'module.' prefix if checkpoint was saved without DataParallel
                new_state_dict = {'module.' + k: v for k, v in state_dict.items()}
                network.load_state_dict(new_state_dict)
            else:
                network.load_state_dict(state_dict)
        else:
            # Remove 'module.' prefix if loading from DataParallel checkpoint to single GPU
            if any(k.startswith('module.') for k in state_dict.keys()):
                new_state_dict = {k[7:]: v for k, v in state_dict.items() if k.startswith('module.')}
                network.load_state_dict(new_state_dict)
            else:
                network.load_state_dict(state_dict)
    else:
        raise KeyError(f"No model loaded {ckpt_path}")
        sys.exit()
        
    if 'collate' in conf.dataset.keys():
        collate_fn = collate_fcs[conf.dataset.collate.type]
    else:
        collate_fn = collate_fcs['base']

    cov_result, rmse = [], []
    net_out_result = {}
    evals = {}
    
    # Set window_size and step_size
    # For online inference, use smaller step_size than window_size
    step_size = args.stepsize if args.stepsize is not None else args.seqlen
    dataset_conf.data_list[0]["window_size"] = args.seqlen
    dataset_conf.data_list[0]["step_size"] = step_size
    
    print(f"[INFO] Window size: {args.seqlen}, Step size: {step_size}")
    if step_size < args.seqlen:
        print(f"[INFO] Online mode: overlapping windows with {args.seqlen - step_size} frame overlap")
    for data_conf in dataset_conf.data_list:
        # Auto-discover files if data_drive is empty
        if data_conf.name == "nymeria" or data_conf.name == "aria" or data_conf.name == "ariaCPF" or data_conf.name == "aria_demo":
            data_drive_list = data_conf.data_drive
            if not data_drive_list:
                import glob
                if data_conf.name == "aria":
                    pickle_files = sorted(glob.glob(os.path.join(data_conf.data_root, "*.pickle")))
                else:
                    pickle_files = sorted(glob.glob(os.path.join(data_conf.data_root, "*.pkl")))
                data_drive_list = [os.path.splitext(os.path.basename(f))[0] for f in pickle_files]
                # data_conf.data_root = '$DATA_ROOT/processed_nymeria_2imu_both/test'
                # data_drive_list = ['20230607_s0_james_johnson_act3_ifj2gc_cpfbody_200hz']
                # print(f"Auto-discovered {len(data_drive_list)} files: {data_drive_list[:3]}...")
                
        elif data_conf.name == "tlioCPF":
            import glob
            import os
            # Look for .npz files instead of directories
            data_drive_list = []
            npz_files = sorted(glob.glob(os.path.join(data_conf.data_root, "*.npz")))
            from tqdm.auto import tqdm as tqdm_
            for npz_path in tqdm_(npz_files, desc="Indexing npz files", unit="file", total=len(npz_files)):
                # Extract filename without extension
                data_name = os.path.splitext(os.path.basename(npz_path))[0]
                # # Remove _cpf suffix if present
                # if data_name.endswith('_cpf'):
                #     data_name = data_name[:-4]
                data_drive_list.append(data_name)
        elif data_conf.name == "inrtl":
            import glob
            data_drive_list = []
            npz_files = sorted(glob.glob(os.path.join(data_conf.data_root, "*_cpf.npz")))
            for npz_path in npz_files:
                data_name = os.path.splitext(os.path.basename(npz_path))[0]
                if data_name.endswith("_cpf"):
                    data_name = data_name[:-4]
                data_drive_list.append(data_name)
            print(f"Auto-discovered {len(data_drive_list)} INRTL sequences: {data_drive_list}")
        # elif data_conf.name == "tlio":
        #     # TLIO uses .npz files, not directories
        #     data_drive_list = data_conf.data_drive
        #     if not data_drive_list:
        #         import glob
        #         npz_files = sorted(glob.glob(os.path.join(data_conf.data_root, "*.npz")))
        #         data_drive_list = [os.path.splitext(os.path.basename(f))[0] for f in npz_files]
        #         print(f"Auto-discovered {len(data_drive_list)} TLIO files: {data_drive_list[:3]}...")
        
        # for path in data_drive_list:
        for path in tqdm.tqdm(data_drive_list, desc="Processing", unit="file"):
            if args.whole:
                dataset_conf["mode"] = "inference"
                # dataset_conf["mode"] = "infevaluate"
            else:
                dataset_conf["mode"] = "infevaluate"
            dataset_conf["exp_dir"] = conf.general.exp_dir
            # eval_dataset = SeqeuncesMotionDataset(data_set_config=dataset_conf, data_path=path, data_root=data_conf["data_root"])    
            eval_dataset = SeqeuncesMotionDataset(data_set_config=dataset_conf, data_path=path, data_root=data_conf.data_root) 
            # Set number of workers for DataLoader (important for multi-GPU)
            num_workers = 4 * torch.cuda.device_count() if args.multi_gpu else 4
            
            eval_loader = Data.DataLoader(dataset=eval_dataset, batch_size=args.batch_size, 
                                            shuffle=False, collate_fn=collate_fn, drop_last = False,
                                            num_workers=num_workers,
                                            pin_memory=True if args.device.startswith('cuda') else False)
            inference_state = inference(
                network=network, 
                loader=eval_loader, 
                confs=conf.train, 
                pose_predictor=pose_predictor,
                step_size=step_size,
                window_size=args.seqlen
            )    
            if not "cov" in inference_state.keys():
                    inference_state["cov"] = torch.zeros_like(inference_state["net_vel"])         
            inference_state['ts'] = inference_state['ts']
            inference_state['net_vel'] = inference_state['net_vel'][0] #TODO: batch size != 1
            # Drop big optional fields if present
            inference_state.pop('joint_pos', None)
            inference_state.pop('joint_rot', None)

            # Move everything you keep to CPU + float32
            for key in ("cov", "net_vel", "ts", "wrapper_y"):
                if key in inference_state and isinstance(inference_state[key], torch.Tensor):
                    inference_state[key] = (
                        inference_state[key]
                        .detach()
                        .to('cpu', dtype=torch.float32, non_blocking=True)
                        .contiguous()
                    )
            net_out_result[path] = inference_state

            # free references
            del inference_state
            torch.cuda.empty_cache()  # optional; frees cached blocks for reuse

    net_result_path = os.path.join(conf.general.exp_dir, 'net_output.pickle')
    print("save netout, ", net_result_path)
    with open(net_result_path, 'wb') as handle:
        pickle.dump(net_out_result, handle, protocol=pickle.HIGHEST_PROTOCOL)
   
   