import argparse
import os
import tqdm, wandb
from model import net_dict

from datasets import collate_fcs, SeqeuncesMotionDataset
import pickle

import numpy as np
import pypose as pp
import torch
import torch.utils.data as Data

import yaml
from pyhocon import ConfigFactory
from pyhocon import HOCONConverter as conf_convert
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.dual_imu_preprocessing import preprocess_dual_imu

from model.losses import get_motion_loss, get_motion_RMSE, get_joint_loss
from utils import (cat_state, move_to, save_ckpt, save_state,
                   write_wandb)
import random

def train(network, loader, confs, epoch, optimizer, pose_predictor=None):
    """
    Training loop for RayNeo dataset
    
    Training setup:
    - Input: glasses IMU (drift-prone) + glasses rotations (drift-prone) 
    - Target: velocities_aligned (drift-free from iPhone SLAM)
    - Goal: Learn to predict clean velocities from drifty sensor inputs
    """
    network.train()
    losses, pred_cov, vel_l2, position_loss, final_position_loss, joint_rot_loss, joint_pos_loss = 0, 0, 0, 0, 0, 0, 0
    
    accumulation_steps = getattr(confs, 'accumulation_steps', 1)
    gradient_clip = getattr(confs, 'gradient_clip', None)
    
    # Check if we should use raw quaternions and dual-IMU mode
    use_raw_quat = getattr(confs, 'use_raw_quat', False)
    dual_imu_mode = getattr(confs, 'dual_imu_mode', 'right_only')

    t_range = tqdm.tqdm(loader)
    for i, (data,_, label) in enumerate(t_range):
        data, label = move_to([data, label], confs.device)

        # Preprocess dual-IMU data based on mode
        data, rot = preprocess_dual_imu(data, mode=dual_imu_mode)

        # For modes that don't modify rotation, extract from label
        if dual_imu_mode in ['right_only', 'average', 'concat_body_no_left_rot',
                             'avg_plus_diff_cross_frame', 'weighted_concat_body']:
            # Extract rotation input (glasses rotations with drift)
            # Choose representation based on config
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

        data["gt_rot"] = pp.SO3(label['gt_rot'])

        noise_sigma = 0.03
        for key in ['init_vel', 'acc', 'gyro', 'right_hip_acc', 'right_hip_rot', 'right_elbow_acc', 'right_elbow_rot', 'rot', 'mobileposer_input']:# mobileposer_input: [N,60]
            if key in data.keys():
                data[key] = data[key] + noise_sigma * torch.randn_like(data[key])

        if pose_predictor is not None:
            # Handle dual-encoder dict structure
            if isinstance(data, dict) and 'cpf' in data:
                T = data['cpf']['acc'].shape[1]
                acc_data = data['cpf']['acc']
                gyro_data = data['cpf']['gyro']
            else:
                T = data['acc'].shape[1]
                acc_data = data['acc']
                gyro_data = data['gyro']

            with torch.no_grad():
                if confs.get('resample_hz', None) is not None:
                    resample_factor = int(confs.resample_hz/50)
                    acc_resampled = acc_data[:, ::resample_factor, :]  # Downsample to target Hz
                    gyro_resampled = gyro_data[:, ::resample_factor, :]  # Downsample to target Hz
                else:
                    resample_factor = 4  # Default: 200Hz -> 50Hz
                    acc_resampled = acc_data[:, ::resample_factor, :]  # Downsample to 50Hz
                    gyro_resampled = gyro_data[:, ::resample_factor, :]  # Downsample to 50Hz
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
                joint_rot = joint_rot + torch.randn_like(joint_rot) * 0.05
            inte_state = network(data, rot, joint_rot=joint_rot) # 200Hz
        else:
            inte_state = network(data, rot)
            
        # Get ground truth labels (velocities_aligned - drift-free targets)
        # Handle DataParallel wrapper
        if hasattr(network, 'module'):
            gt_label = network.module.get_label(label['gt_vel'])  # Clean velocity targets
        else:
            gt_label = network.get_label(label['gt_vel'])  # Clean velocity targets
            
        # Pass rot for gravity decomposition loss
        data_with_rot = {**data, 'rot': rot}
        loss_state = get_motion_loss(inte_state, gt_label, confs, data_with_rot)

        if "joint_rot" in inte_state.keys() and "joint_pos" in inte_state.keys():
            # Handle DataParallel wrapper for joint labels
            if hasattr(network, 'module'):
                gt_joint_pos = network.module.get_label(data['joint_pos'])
                gt_joint_rot = network.module.get_label(data['joint_rot'])
            else:
                gt_joint_pos = network.get_label(data['joint_pos'])
                gt_joint_rot = network.get_label(data['joint_rot'])
            joint_loss_state = get_joint_loss(inte_state, gt_joint_pos, gt_joint_rot, confs)
            joint_rot_loss += joint_loss_state["joint_rot_loss"]
            joint_pos_loss += joint_loss_state["joint_pos_loss"]
            
            # Scale loss by accumulation steps
            loss = (loss_state["loss"] + joint_loss_state["loss"]) / accumulation_steps
        else:
            # Scale loss by accumulation steps
            loss = loss_state["loss"] / accumulation_steps

        # statistics
        losses += loss_state["loss"].item()
        vel_l2 += loss_state["vel_l2"].item()
        position_loss += loss_state["position_loss"].item()
        final_position_loss += loss_state["final_position_loss"].item()
            
        if confs.propcov:
            pred_cov += loss_state["cov_loss"].mean().item()

        t_range.set_description(
            f"training epoch: %03d,losses: %.06f" % (epoch, loss_state["loss"])
        )

        t_range.refresh()
        
        # Accumulate gradients
        loss.backward()
        
        # Update weights only every accumulation_steps
        if (i + 1) % accumulation_steps == 0:
            # Gradient clipping (NEW)
            if gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(network.parameters(), gradient_clip)
            
            optimizer.step()
            optimizer.zero_grad()

    # Handle remaining gradients
    if (i + 1) % accumulation_steps != 0:
        # Gradient clipping (NEW)
        if gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(network.parameters(), gradient_clip)
            
        optimizer.step()
        optimizer.zero_grad()

    if getattr(confs, 'joint_rot_weight', 0.0) != 0.0:
        return {"loss": (losses / (i + 1)), "cov": (pred_cov / (i + 1)), "vel_l2": (vel_l2 / (i + 1)), "position_loss": (position_loss / (i + 1)), "final_position_loss": (final_position_loss / (i + 1)),
                "joint_rot_loss":(joint_rot_loss / (i + 1)).item(), "joint_pos_loss":(joint_pos_loss / (i + 1)).item()}
    else:
        return {"loss": (losses / (i + 1)), "cov": (pred_cov / (i + 1)), "vel_l2": (vel_l2 / (i + 1)), "position_loss": (position_loss / (i + 1)), "final_position_loss": (final_position_loss / (i + 1))}


def test(network, loader, confs, pose_predictor=None):
    """Modified test function"""
    # CRITICAL FIX: Set network to eval mode for proper testing
    # This disables dropout and uses BatchNorm running statistics
    network.eval()
    use_raw_quat = getattr(confs, 'use_raw_quat', False)
    dual_imu_mode = getattr(confs, 'dual_imu_mode', 'right_only')

    with torch.no_grad():
        losses, pred_cov, vel_l2, position_loss, final_position_loss, joint_rot_loss, joint_pos_loss  = 0, 0, 0, 0, 0, 0, 0

        t_range = tqdm.tqdm(loader)
        for i, (data, _, label) in enumerate(t_range):
            data,label = move_to([data, label], confs.device)

            # Preprocess dual-IMU data based on mode
            data, rot = preprocess_dual_imu(data, mode=dual_imu_mode)

            # For modes that don't modify rotation, extract from label
            if dual_imu_mode in ['right_only', 'average', 'concat_body_no_left_rot',
                                 'avg_plus_diff_cross_frame', 'weighted_concat_body']:
                # Choose representation based on config
                if use_raw_quat:
                    if hasattr(label['gt_rot'], 'quaternion'):
                        rot = label['gt_rot'][:,:-1,:].quaternion()
                    else:
                        rot = label['gt_rot'][:,:-1,:]
                        
                    if hasattr(label['gt_rot'], 'Log'):
                        rot_ = label['gt_rot'][:,:-1,:].Log().tensor()
                    else:
                        # If it's already a regular tensor, assume it's in quaternion format
                        rot_ = pp.SO3(label['gt_rot'][:,:-1,:]).Log().tensor()
                else:
                    if hasattr(label['gt_rot'], 'Log'):
                        rot = label['gt_rot'][:,:-1,:].Log().tensor()
                    else:
                        rot = pp.SO3(label['gt_rot'][:,:-1,:]).Log().tensor()

            data["gt_rot"] = pp.SO3(label['gt_rot'])
            
            if pose_predictor is not None:
                T = data['acc'].shape[1]
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
                
            # Handle DataParallel wrapper
            if hasattr(network, 'module'):
                gt_label = network.module.get_label(label['gt_vel'])
            else:
                gt_label = network.get_label(label['gt_vel'])
            
            # Pass rot for full loss computation (including position loss)
            data_with_rot = {**data, 'rot': rot}
            loss_state = get_motion_loss(inte_state, gt_label, confs, data_with_rot)
            
            if "joint_rot" in inte_state.keys() and "joint_pos" in inte_state.keys():
                # Handle DataParallel wrapper for joint labels
                if hasattr(network, 'module'):
                    gt_joint_pos = network.module.get_label(data['joint_pos'])
                    gt_joint_rot = network.module.get_label(data['joint_rot'])
                else:
                    gt_joint_pos = network.get_label(data['joint_pos'])
                    gt_joint_rot = network.get_label(data['joint_rot'])
                joint_loss_state = get_joint_loss(inte_state, gt_joint_pos, gt_joint_rot, confs)
                joint_rot_loss += joint_loss_state["joint_rot_loss"]
                joint_pos_loss += joint_loss_state["joint_pos_loss"]
            
            # statistics
            losses += loss_state["loss"].item()
            vel_l2 += loss_state["vel_l2"].item()
            position_loss += loss_state["position_loss"].item() 
            final_position_loss += loss_state["final_position_loss"].item() 
            
            if confs.propcov:
                pred_cov += loss_state["cov_loss"].mean().item()
                cov_loss_value = torch.sqrt(loss_state['cov_loss'])
            else:   
                cov_loss_value = 0
            t_range.set_description(
                "testing loss: %.06f, cov: %.06f, vel_l2: %.06f" % (
                    losses / (i + 1), 
                    cov_loss_value, 
                    loss_state['vel_l2']
                )
            )

            t_range.refresh()
            
    if getattr(confs, 'joint_rot_weight', 0.0) != 0.0:
        return {"loss": (losses / (i + 1)), "cov": (pred_cov / (i + 1)), "vel_l2": (vel_l2 / (i + 1)), "position_loss": (position_loss / (i + 1)), "final_position_loss": (final_position_loss / (i + 1)),
                "joint_rot_loss":(joint_rot_loss / (i + 1)).item(), "joint_pos_loss":(joint_pos_loss / (i + 1)).item()}
    else:
        return {"loss": (losses / (i + 1)), "cov": (pred_cov / (i + 1)), "vel_l2": (vel_l2 / (i + 1)), "position_loss": (position_loss / (i + 1)), "final_position_loss": (final_position_loss / (i + 1))}



def evaluate(network, loader, confs, silent_tqdm=False, pose_predictor=None):
    # CRITICAL FIX: Set network to eval mode for proper evaluation
    # This disables dropout and uses BatchNorm running statistics
    network.eval()
    use_raw_quat = getattr(confs, 'use_raw_quat', False)
    dual_imu_mode = getattr(confs, 'dual_imu_mode', 'right_only')
    # Use running averages instead of storing all results
    total_loss = 0.0
    total_cov = 0.0
    count = 0

    with torch.no_grad():
        for i, (data, _, label) in enumerate(tqdm.tqdm(loader, disable=silent_tqdm)):
            data, label = move_to([data, label], confs.device)

            # Preprocess dual-IMU data based on mode
            data, rot = preprocess_dual_imu(data, mode=dual_imu_mode)

            # For modes that don't modify rotation, extract from label
            if dual_imu_mode in ['right_only', 'average', 'concat_body_no_left_rot',
                                 'avg_plus_diff_cross_frame', 'weighted_concat_body']:
                # Choose representation based on config
                if use_raw_quat:
                    if hasattr(label['gt_rot'], 'quaternion'):
                        rot = label['gt_rot'][:,:-1,:].quaternion()
                    else:
                        rot = label['gt_rot'][:,:-1,:]
                    
                    if hasattr(label['gt_rot'], 'Log'):
                        rot_ = label['gt_rot'][:,:-1,:].Log().tensor()
                    else:
                        # If it's already a regular tensor, assume it's in quaternion format
                        rot_ = pp.SO3(label['gt_rot'][:,:-1,:]).Log().tensor()
                else:
                    if hasattr(label['gt_rot'], 'Log'):
                        rot = label['gt_rot'][:,:-1,:].Log().tensor()
                    else:
                        rot = pp.SO3(label['gt_rot'][:,:-1,:]).Log().tensor()

            data["gt_rot"] = pp.SO3(label['gt_rot'])
            
            if pose_predictor is not None:
                T = data['acc'].shape[1]
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
            
            # Handle DataParallel wrapper
            if hasattr(network, 'module'):
                gt_label = network.module.get_label(label['gt_vel'])
            else:
                gt_label = network.get_label(label['gt_vel'])
                
            loss_state = get_motion_RMSE(inte_state, gt_label, confs)                
            # Accumulate statistics without storing tensors
            total_loss += loss_state["loss"].item()
            if "cov" in inte_state and inte_state["cov"] is not None and confs.propcov:
                total_cov += inte_state["cov"].mean().item()
            count += 1
            
            # Clean up GPU memory immediately
            del loss_state, inte_state, data, label, rot, gt_label
            if i % 10 == 0:
                torch.cuda.empty_cache()
        
    avg_loss = total_loss / count if count > 0 else 0.0
    avg_cov = total_cov / count if count > 0 else 0.0
    
    print("evaluating: vel losses %f, evaluation cov %f" % (avg_loss, avg_cov))
    
    return {
        "loss": {"loss": torch.tensor(avg_loss)},
        "cov": avg_cov
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/EuRoC/motion_body.conf",
        help="config file path",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="cuda or cpu, Default is cuda:0"
    )
    parser.add_argument(
        "--gpu", type=int, help="GPU ID to use (e.g., 0, 1, 2). Overrides --device if specified"
    )
    parser.add_argument(
        "--multi_gpu", default=False, action="store_true", 
        help="Use all available GPUs with DataParallel"
    )
    parser.add_argument(
        "--gpu_ids", type=str, default=None,
        help="Comma-separated GPU IDs to use (e.g., '0,1,2'). Used with --multi_gpu"
    )
    parser.add_argument(
        "--load_ckpt",
        default=False,
        action="store_true",
        help="If True, try to load the newest.ckpt in the \
                                                                                exp_dir specificed in our config file.",
    )
    parser.add_argument(
        "--log",
        default=True,
        action="store_false",
        help="if True, save the meta data with wandb",
    )
    parser.add_argument(
        "--finetune",
        default=False,
        action="store_true",
        help="Enable fine-tuning mode - saves models in experiments/finetuned/ directory"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )

    args = parser.parse_args()

    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # if using multi-GPU
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {args.seed} for reproducibility")
    
    # Handle GPU configuration
    if args.device == "cpu":
        gpu_ids = None
        args.multi_gpu = False
        print(f"Using CPU device (forced)")
    elif args.multi_gpu:
        if args.gpu_ids:
            gpu_ids = [int(id.strip()) for id in args.gpu_ids.split(',')]
            args.device = f"cuda:{gpu_ids[0]}"
            print(f"Using multiple GPUs: {gpu_ids}")
        else:
            gpu_ids = list(range(torch.cuda.device_count()))
            args.device = "cuda:0"
            print(f"Using all available GPUs: {gpu_ids}")
    elif args.gpu is not None:
        args.device = f"cuda:{args.gpu}"
        gpu_ids = None
        print(f"Using single GPU {args.gpu} (device: {args.device})")
    else:
        gpu_ids = None
    
    print(args)
    conf = ConfigFactory.parse_file(args.config)

    # Force device setting - args.device takes precedence
    conf.train.device = args.device
    print(f"Final device setting: {args.device}")
    print(f"Training batch size: {conf.train.batch_size}")
    exp_folder = os.path.split(conf.general.exp_dir)[-1]
    conf_name = os.path.split(args.config)[-1].split(".")[0]
    
    # Modify exp_dir for fine-tuning
    if args.finetune:
        # Use the exp_dir as-is if it already contains "finetuned", otherwise add it
        if "finetuned" not in conf.general.exp_dir:
            base_exp_dir = conf.general.exp_dir.replace("experiments/", "experiments/finetuned/")
        else:
            base_exp_dir = conf.general.exp_dir
        conf["general"]["exp_dir"] = os.path.join(base_exp_dir, conf_name)
        print(f"Fine-tuning mode: saving models to {conf.general.exp_dir}")
    else:
        # Check if the exp_dir already ends with the config name to avoid duplication
        if not conf.general.exp_dir.endswith(conf_name):
            conf["general"]["exp_dir"] = os.path.join(conf.general.exp_dir, conf_name)
        # else keep exp_dir as is
    if "gravity" in conf.dataset.train:
        gravity = conf.dataset.train.gravity
        conf.train.put("gravity", conf.dataset.train.gravity)
    else:
        gravity = 9.81007

    train_dataset = SeqeuncesMotionDataset(data_set_config=conf.dataset.train)
    test_dataset = SeqeuncesMotionDataset(data_set_config=conf.dataset.test)
    eval_dataset = SeqeuncesMotionDataset(data_set_config=conf.dataset.eval)

    if "collate" in conf.dataset.keys():
        collate_fn_train, collate_fn_test = collate_fcs[conf.dataset.collate.type], collate_fcs[conf.dataset.collate.type]
    else:
        collate_fn_train, collate_fn_test = collate_fcs["base"], collate_fcs["base"]

    
    # Set number of workers for DataLoader (important for multi-GPU)
    # if args.device == "cpu":
    #     num_workers = 0  # No multiprocessing on CPU to avoid issues
    # else:
    #     num_workers = 4 * torch.cuda.device_count() if args.multi_gpu else 4
    num_workers = 0
    
    # Create generator with fixed seed for DataLoader
    generator = torch.Generator()
    generator.manual_seed(args.seed)

    train_loader = Data.DataLoader(
        dataset=train_dataset,
        batch_size=conf.train.batch_size,
        shuffle = True,
        collate_fn=collate_fn_train,
        num_workers=num_workers,
        pin_memory=True if args.device.startswith('cuda') else False,
        generator=generator,  # Use seeded generator for reproducible shuffling
        worker_init_fn=lambda worker_id: np.random.seed(args.seed + worker_id)  # Seed workers
    )
    test_loader = Data.DataLoader(
        dataset=test_dataset,
        batch_size=conf.train.batch_size,
        shuffle=False,
        collate_fn=collate_fn_test,
        num_workers=num_workers,
        pin_memory=True if args.device.startswith('cuda') else False,
        worker_init_fn=lambda worker_id: np.random.seed(args.seed + worker_id)  # Seed workers
    )
    eval_loader = Data.DataLoader(
        dataset=eval_dataset,
        batch_size=conf.train.batch_size,  # Use training batch size for faster evaluation
        shuffle=False,
        collate_fn=collate_fn_test,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True if args.device.startswith('cuda') else False,
        worker_init_fn=lambda worker_id: np.random.seed(args.seed + worker_id)  # Seed workers
    )

    os.makedirs(os.path.join(conf.general.exp_dir, "ckpt"), exist_ok=True)
    with open(os.path.join(conf.general.exp_dir, "parameters.yaml"), "w") as f:
        f.write(conf_convert.to_yaml(conf))

    if not args.log:
        wandb.disabled = True
        print("wandb is disabled")
    else:
        wandb.init(
            project="AirIO" + exp_folder,
            config=conf.train,
            group=conf.train.network,
            name=conf_name,
        )

    ## optimizer and network
    network = net_dict[conf.train.network](conf.train).to(
        device=args.device, dtype=train_dataset.get_dtype()
    )
    if conf.get('prior', None) is not None:
        pose_predictor = net_dict[conf.prior.network](conf.prior).to(
        device=args.device, dtype=train_dataset.get_dtype()
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

    # Support differential learning rates for fine-tuning
    actual_network = network.module if isinstance(network, torch.nn.DataParallel) else network

    if hasattr(actual_network, 'freeze_body') and not actual_network.freeze_body:
        # Fine-tuning mode: use differential learning rates
        body_lr = getattr(conf.train, 'body_lr', conf.train.lr / 10)  # Default: 10x lower
        wrapper_lr = conf.train.lr

        param_groups = [
            {'params': actual_network.body_model.parameters(), 'lr': body_lr, 'name': 'body'},
            {'params': [p for n, p in actual_network.named_parameters()
                       if 'body_model' not in n], 'lr': wrapper_lr, 'name': 'wrapper'}
        ]

        optimizer = torch.optim.Adam(param_groups, weight_decay=conf.train.weight_decay)
        print(f"✓ Using differential learning rates: body={body_lr:.2e}, wrapper={wrapper_lr:.2e}")
    else:
        # Standard mode: single learning rate for all parameters
        optimizer = torch.optim.Adam(
            network.parameters(), lr=conf.train.lr, weight_decay=conf.train.weight_decay
        )  # to use with ViTs
    scheduler = ReduceLROnPlateau(
        optimizer,
        "min",
        factor=conf.train.factor,
        patience=conf.train.patience,
        min_lr=conf.train.min_lr,
    )
    best_loss = np.inf
    epoch = 0

    ## load the chkp if there exist
    if args.load_ckpt:
        if os.path.isfile(os.path.join(conf.general.exp_dir, "ckpt/newest.ckpt")):
            checkpoint = torch.load(
                os.path.join(conf.general.exp_dir, "ckpt/newest.ckpt"),
                map_location=args.device,
                weights_only=True
            )
            # Handle DataParallel checkpoint loading
            if args.multi_gpu and torch.cuda.device_count() > 1:
                # If saved model was DataParallel but current isn't, or vice versa
                state_dict = checkpoint["model_state_dict"]
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith('module.'):
                        # Remove 'module.' prefix if present
                        new_state_dict[k[7:]] = v
                    else:
                        # Add 'module.' prefix if not present
                        new_state_dict['module.' + k] = v
                try:
                    network.load_state_dict(new_state_dict)
                except:
                    # If that fails, try the original state dict
                    network.load_state_dict(state_dict)
            else:
                # Handle single GPU loading
                state_dict = checkpoint["model_state_dict"]
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith('module.'):
                        # Remove 'module.' prefix if loading from DataParallel checkpoint
                        new_state_dict[k[7:]] = v
                    else:
                        new_state_dict[k] = v
                network.load_state_dict(new_state_dict)
            
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            epoch = checkpoint["epoch"]
            best_loss = checkpoint["best_loss"]
            print(
                "loaded state dict %s best_loss %f"
                % (os.path.join(conf.general.exp_dir, "ckpt/newest.ckpt"), best_loss)
            )
            
            # Reset epoch for fine-tuning to start fresh
            if args.finetune:
                print(f"Fine-tuning mode: resetting epoch from {epoch} to 0")
                epoch = 0
                best_loss = np.inf  # Reset best loss for fine-tuning
        else:
            print("Can't find the checkpoint")

    for epoch_i in range(epoch, conf.train.max_epoches):
        train_loss = train(network, train_loader, conf.train, epoch_i, optimizer, pose_predictor=pose_predictor)
        test_loss = test(network, test_loader, conf.train, pose_predictor=pose_predictor)
        print("train loss: %f test loss: %f" % (train_loss["loss"], test_loss["loss"]))

        # save the training meta information
        if args.log:
            write_wandb("train", train_loss, epoch_i)
            write_wandb("test", test_loss, epoch_i)
            # Log learning rates for all parameter groups (support differential LR)
            if len(scheduler.optimizer.param_groups) > 1:
                for i, group in enumerate(scheduler.optimizer.param_groups):
                    group_name = group.get('name', f'group_{i}')
                    write_wandb(f"lr/{group_name}", group["lr"], epoch_i)
            else:
                write_wandb("lr", scheduler.optimizer.param_groups[0]["lr"], epoch_i)
        if epoch_i % conf.train.eval_freq == conf.train.eval_freq - 1:
            eval_state = evaluate(network=network, loader=eval_loader, confs=conf.train, pose_predictor=pose_predictor)
            if args.log:
                write_wandb('eval/loss', eval_state['loss']['loss'], epoch_i)
            if "supervise_pos" in conf.train:
                print("eval pos: %f "%(eval_state['loss']['loss']))
            else:
                print("eval vel: %f "%(eval_state['loss']['loss']))

        scheduler.step(test_loss["loss"])
        if test_loss["loss"] < best_loss:
            best_loss = test_loss["loss"]
            save_best = True
        else:
            save_best = False

        save_ckpt(
            network,
            optimizer,
            scheduler,
            epoch_i,
            best_loss,
            conf,
            save_best=save_best,
        )

    wandb.finish()