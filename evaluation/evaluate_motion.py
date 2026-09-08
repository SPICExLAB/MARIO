# output the trajectory in the world frame for visualization and evaluation
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

import os
import json
import argparse
import numpy as np
import pypose as pp
from datetime import datetime

import torch
import torch.utils.data as Data

from pyhocon import ConfigFactory
from datasets import imu_seq_collate,SeqDataset
 
from utils import CPU_Unpickler, integrate, interp_xyz
from utils.velocity_integrator import Velocity_Integrator, integrate_pos, integrate_pos_with_orientation, integrate_pos_world_frame

from utils.visualize_state import visualize_3D_motion_with_xz, visualize_2d_trajectory
import pickle

# Import MSCKF filter
try:
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'EKF'))
    from msckf_filter import MSCKFRunner
    MSCKF_AVAILABLE = True
except ImportError as e:
    print(f"Warning: MSCKF not available: {e}")
    MSCKF_AVAILABLE = False
 
def calculate_rte(outstate, duration, step_size):
    poses = outstate['poses']  # Shape: [1, N-1, 3] 
    poses_gt = outstate['poses_gt']  # Shape: [1, N-1, 3]
    
    # Ensure same length
    min_len = min(poses.shape[1], poses_gt.shape[1])
    poses = poses[:, :min_len, :]
    poses_gt = poses_gt[:, :min_len, :]
    
    # Calculate relative displacements
    if min_len > duration:
        dp = poses[:, duration-1:] - poses[:, :-duration+1]
        dp_gt = poses_gt[:, duration-1:] - poses_gt[:, :-duration+1]
        rte = (dp - dp_gt).norm(dim=-1)
    else:
        rte = torch.tensor([])
    return rte

def match_nearest_two_pointer(gt_ts: torch.Tensor, vel_ts: torch.Tensor) -> torch.Tensor:
    gt = gt_ts.contiguous().view(-1)
    v  = vel_ts.contiguous().view(-1)
    i = 0
    out = torch.empty(v.numel(), dtype=torch.long)

    for k in range(v.numel()):
        # advance while the next gt time is closer
        while i + 1 < gt.numel() and abs(v[k] - gt[i+1]) <= abs(v[k] - gt[i]):
            i += 1
        out[k] = i
        # optional look-ahead to reduce backtracking:
        if k + 1 < v.numel() and v[k+1] >= gt[i] and i + 1 < gt.numel():
            # advance minimally to stay near track
            while i + 1 < gt.numel() and gt[i+1] <= v[k+1]:
                i += 1
    return out


def integrate_with_msckf(motion_dataset, inference_state, gravity, args):
    """
    Integrate velocities using Multi-State Constraint Kalman Filter
    """
    print(f"\n🔧 Running MSCKF Integration:")
    print(f"   Sliding window states: Enabled")
    print(f"   Clone state management: Auto-managed")
    
    # Initialize MSCKF filter with proper window size - FIXED: get from args properly
    window_size = args.msckf_window if hasattr(args, 'msckf_window') else 250
    slide_step = args.msckf_slide if hasattr(args, 'msckf_slide') else 50
    max_clones = args.msckf_clones if hasattr(args, 'msckf_clones') else None
    
    print(f"   MSCKF parameters: window={window_size}, slide={slide_step}, clones={max_clones}")
    
    msckf_runner = MSCKFRunner(
        gravity=gravity, 
        window_size=window_size,
        slide_step=slide_step,
        max_clone_states=max_clones
    )
    
    # Get initial conditions - CORRECTED: keep velocity in body frame for proper MSCKF initialization
    initial_vel_body = motion_dataset.data['velocity'][0]  # Body frame velocity
    initial_orientation = motion_dataset.data['gt_orientation'][0]  # CPF→World rotation
    
    init_data = {
        'pos': torch.zeros(3, dtype=torch.float64),
        'rot': initial_orientation,
        'vel': initial_vel_body.double()  # Keep in body frame - MSCKF will transform internally
    }
    
    msckf_runner.initialize(init_data['pos'], init_data['rot'], init_data['vel'])
    
    # Get synchronized data
    gt_ts = motion_dataset.data['time']
    vel_ts = inference_state['ts']
    
    # Fix timestamp shape
    if len(vel_ts.shape) == 3:
        vel_ts = vel_ts.squeeze(0)
    elif len(vel_ts.shape) == 1:
        vel_ts = vel_ts.unsqueeze(-1)
    
    # Vectorized timestamp matching
    vel_times = vel_ts[:, 0]
    gt_times = gt_ts.unsqueeze(0)  # Shape: [1, N_gt]
    vel_times_expanded = vel_times.unsqueeze(1)  # Shape: [N_vel, 1]
    
    # Compute all differences at once
    time_diffs = torch.abs(gt_times - vel_times_expanded)  # Shape: [N_vel, N_gt]
    velocity_indices = torch.argmin(time_diffs, dim=1).numpy()  # Shape: [N_vel]
    
    # Interpolate network velocities to all timesteps 
    gt_ts_np = gt_ts.numpy()
    vel_ts_np = vel_ts[:, 0].numpy()
    net_vel_np = inference_state['net_vel'].numpy()
    
    from utils import interp_xyz
    net_vel_result = interp_xyz(gt_ts_np, vel_ts_np, net_vel_np)
    if isinstance(net_vel_result, torch.Tensor):
        net_vel_interp = net_vel_result.float()
    else:
        net_vel_interp = torch.from_numpy(net_vel_result).float()

    
    print(f"   Processing {len(gt_ts)} IMU samples with {len(velocity_indices)} velocity observations")
    
    # Get ground truth positions
    if 'gt_pos' in motion_dataset.data:
        gt_positions = motion_dataset.data['gt_pos']
    elif 'position' in motion_dataset.data:
        gt_positions = motion_dataset.data['position']
    else:
        # Vectorized integration from velocities
        # For MSCKF: integrate from world frame velocities to match MSCKF trajectory frame
        velocities_body = motion_dataset.data['velocity']
        gt_orientations = motion_dataset.data['gt_orientation']
        R_cpf_to_world = gt_orientations.matrix().float()  # Shape: [N, 3, 3]
        velocities_world = torch.bmm(R_cpf_to_world, velocities_body.unsqueeze(-1)).squeeze(-1)
        
        dts = torch.diff(gt_ts, prepend=torch.tensor([0.0]))
        displacements = velocities_world * dts.unsqueeze(1)
        gt_positions = torch.cumsum(displacements, dim=0)
    
    # Pre-compute all dt values
    dts = torch.diff(gt_ts, prepend=gt_ts[0:1])
    
    # Pre-extract IMU data
    gyro_data = motion_dataset.data['gyro']
    acc_data = motion_dataset.data['acc']
    gt_vel_data = motion_dataset.data['velocity']  # Extract ground truth velocities
    
    # Handle covariance
    if 'cov' in inference_state:
        covs = inference_state['cov'].numpy()
        covs = np.maximum(covs, 0.001)
    else:
        covs = np.full((len(vel_ts), 3), 0.01)
    
    # Process in batches for efficiency
    trajectory_positions = []
    vel_obs_idx = 0
    
    # Process all IMU data and velocity observations
    trajectory_velocities = []  # Add velocity tracking
    for i in range(1, min(len(gt_ts), len(gt_positions))):
        # Add IMU measurement
        imu_data = {
            "gyro": gyro_data[i],
            "acc": acc_data[i],
            "dt": dts[i].item()
        }
        msckf_runner.add_imu_data(imu_data)
        
        # Check for velocity observation
        if vel_obs_idx < len(velocity_indices) and i >= velocity_indices[vel_obs_idx]:
            # Use the correct velocity index, not IMU index
            vel_idx = min(velocity_indices[vel_obs_idx], len(net_vel_interp) - 1)
            state = msckf_runner.add_velocity_observation(
                net_vel_interp[vel_idx], 
                covs[vel_obs_idx]
            )
            if state is not None:
                trajectory_positions.append(state['position'])
                trajectory_velocities.append(state['velocity'])  # Store velocity
            vel_obs_idx += 1
        else:
            if msckf_runner.current_state is not None:
                trajectory_positions.append(msckf_runner.current_state['position'])
                trajectory_velocities.append(msckf_runner.current_state['velocity'])  # Store velocity
            else:
                trajectory_positions.append(
                    trajectory_positions[-1] if trajectory_positions else np.zeros(3)
                )
                trajectory_velocities.append(
                    trajectory_velocities[-1] if trajectory_velocities else np.zeros(3)
                )
    
    # Vectorized conversion to tensors
    if trajectory_positions:
        poses = torch.tensor(np.array(trajectory_positions), dtype=torch.float32).unsqueeze(0)
        velocities = torch.tensor(np.array(trajectory_velocities), dtype=torch.float32)
        
        # Vectorized ground truth relative positions
        gt_pos_relative = gt_positions[1:len(trajectory_positions)+1] - gt_positions[0]
        poses_gt = gt_pos_relative.unsqueeze(0)
        
        # Compute velocity error (compare with ground truth velocities)
        gt_velocities = gt_vel_data[1:len(trajectory_velocities)+1]
        
        # Transform GT velocities to world frame for proper comparison with MSCKF
        # MSCKF produces world frame velocities, but GT is in body/CPF frame
        gt_orientations = motion_dataset.data['gt_orientation'][1:len(trajectory_velocities)+1]
        R_cpf_to_world = gt_orientations.matrix().float()  # Shape: [N, 3, 3]
        gt_velocities_world = torch.bmm(R_cpf_to_world, gt_velocities.unsqueeze(-1)).squeeze(-1)
        
        vel_dist = (velocities - gt_velocities_world).norm(dim=-1)
        
        # Vectorized distance calculation
        pos_dist = (poses[0] - poses_gt[0]).norm(dim=-1)
        
        # Calculate trajectory length (sum of step distances)
        pred_steps = torch.diff(poses[0], dim=0)
        pred_trajectory_length = pred_steps.norm(dim=1).sum().item()
        
        gt_steps = torch.diff(poses_gt[0], dim=0)
        gt_trajectory_length = gt_steps.norm(dim=1).sum().item()
        
        print(f"   MSCKF trajectory generated: {poses.shape[1]} poses")
        print(f"   Predicted trajectory length: {pred_trajectory_length:.2f}m")
        print(f"   GT trajectory length: {gt_trajectory_length:.2f}m")
        print(f"   Final position error: {pos_dist[-1]:.3f}m")
        
        return {
            'poses': poses,
            'poses_gt': poses_gt,
            'vel_dist': vel_dist,
            'pos_dist': pos_dist
        }
    
    return None
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu", help="cuda or cpu")
    parser.add_argument("--exp", type=str, default="experiments/euroc/motion_body", help="Path for AirIO netoutput")
    parser.add_argument("--seqlen", type=int, default="1000", help="the length of the segment")
    parser.add_argument("--dataconf", type=str, default="configs/datasets/EuRoC/Euroc_body.conf", help="the configuration of the dataset")
    parser.add_argument("--savedir",type=str,default = "./result/inrtl",help = "Directory where the results will be saved")
    parser.add_argument("--usegtrot", action="store_true", help="Use ground truth rotation for gravity compensation")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument("--use_msckf", action="store_true", help="Use MSCKF for sliding window velocity integration")
    parser.add_argument("--msckf_window", type=int, default=250, help="MSCKF window size")
    parser.add_argument("--msckf_slide", type=int, default=50, help="MSCKF slide step")
    parser.add_argument("--msckf_clones", type=int, default=None, help="MSCKF max clone states (auto if None)")
    parser.add_argument("--vis", action="store_true", help="Enable 3D visualization of trajectories")

    args = parser.parse_args(); 
    print(("\n"*3) + str(args) + ("\n"*3))
    config = ConfigFactory.parse_file(args.dataconf)
    # Try dataset.inference first (if dataconf is wrapped), fall back to inference (if direct)
    try:
        dataset_conf = config.get_config('dataset.inference')
    except:
        dataset_conf = config.get_config('inference')
    print(dataset_conf.keys())

    if args.exp is not None:
        net_result_path = os.path.join(args.exp, 'net_output.pickle')
        if os.path.isfile(net_result_path):
            with open(net_result_path, 'rb') as handle:
                inference_state_load = CPU_Unpickler(handle).load()
                updated_inference_state = inference_state_load.copy() # dict_keys(['cov', 'net_vel', 'ts'])
        else:
            raise Exception(f"Unable to load the network result: {net_result_path}")
    
    # Extract config file name without path and extension
    config_name = os.path.splitext(os.path.basename(args.dataconf))[0]

    # Extract experiment directory structure from --exp path
    # Example: "experiments/aria_world_fusion/wrapper_ablation/9_fusion_direct"
    #       -> "aria_world_fusion/wrapper_ablation/9_fusion_direct"
    exp_path_parts = args.exp.split('/')
    if 'experiments' in exp_path_parts:
        # Get everything after 'experiments' directory
        exp_idx = exp_path_parts.index('experiments')
        exp_structure = '/'.join(exp_path_parts[exp_idx + 1:])
    else:
        # Fallback: use the last directory name from exp path
        exp_structure = os.path.basename(args.exp)

    # Add method suffix to folder name
    method_suffix = "msckf" if (args.use_msckf and MSCKF_AVAILABLE) else "standard"

    # Build folder path: savedir/exp_structure/config_name/
    # Example: ./result/inrtl/aria_world_fusion/wrapper_ablation/9_fusion_direct/ariaCPF_world_wo_g/
    folder = os.path.join(args.savedir, exp_structure, config_name)
    os.makedirs(folder, exist_ok=True)

    print(f"\n📁 Results will be saved to: {folder}")
    print(f"📊 Integration method: {method_suffix.upper()}")

    AllResults = []
    net_out_result = {}

    for data_conf in dataset_conf.data_list:
        print(data_conf)
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
                print(f"Auto-discovered {len(data_drive_list)} files: {data_drive_list[:3]}...")
                # data_conf.data_root = '$DATA_ROOT/Aria_data_collection_processed'
                # data_drive_list = ['Stairs_cpfbody_200hz']
                
                print(f"Auto-discovered {len(data_drive_list)} files: {data_drive_list[:3]}...")
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
        #     data_drive_list = data_conf.data_drive
        #     if not data_drive_list:
        #         import glob
        #         npz_files = sorted(glob.glob(os.path.join(data_conf.data_root, "*.npz")))
        #         data_drive_list = [os.path.splitext(os.path.basename(f))[0] for f in npz_files]
        #         print(f"Auto-discovered {len(data_drive_list)} files: {data_drive_list[:3]}...")

        
        # for data_name in data_drive_list:
        from tqdm import tqdm
        for data_name in tqdm(data_drive_list, desc="Processing", unit="file"):
            print(f"\n{'='*70}")
            print(f"Processing: {data_name}")
            print(f"{'='*70}")

            dataset = SeqDataset(data_conf.data_root, data_name, args.device, name = data_conf.name, 
                                duration=args.seqlen, step_size=args.seqlen, drop_last=False, conf = dataset_conf)
            loader = Data.DataLoader(dataset=dataset, batch_size=1, collate_fn=imu_seq_collate, 
                                    shuffle=False, drop_last=False)
            init = dataset.get_init_value()
            gravity = dataset.get_gravity()
            
            # Start GT trajectory at origin for relative comparison
            integrator_outstate = pp.module.IMUPreintegrator(
                torch.zeros_like(init['pos']), init['rot'], init['vel'],gravity=gravity,
                reset=False
            ).to(args.device).double()
            
            integrator_reset = pp.module.IMUPreintegrator(
                torch.zeros_like(init['pos']), init['rot'], init['vel'],gravity = gravity,
                reset=True
            ).to(args.device).double()
            
            outstate = integrate(
                integrator_outstate, loader, init, 
                device=args.device, gtinit=False, save_full_traj=True,
                use_gt_rot=args.usegtrot
            )
            
            relative_outstate = integrate(
                integrator_reset, loader, init, 
                device=args.device, gtinit=True,
                use_gt_rot=args.usegtrot
            )
            
            if args.exp is not None:
                motion_dataset = SeqDataset(data_conf.data_root, data_name, args.device, name = data_conf.name, 
                                           duration=args.seqlen, step_size=args.seqlen, drop_last=False, conf = dataset_conf)
                motion_loader = Data.DataLoader(dataset=motion_dataset, batch_size=1, collate_fn=imu_seq_collate, 
                                               shuffle=False, drop_last=False)
                inference_state = inference_state_load[data_name]
                gt_ts =  motion_dataset.data['time']
                vel_ts = inference_state['ts']
                # Fix timestamp shape if needed
                if len(vel_ts.shape) == 1:
                    vel_ts = vel_ts.unsqueeze(-1)
                elif len(vel_ts.shape) == 3:  # [1, N, 1] -> [N, 1]
                    vel_ts = vel_ts.squeeze(0)
                # Synchronize network output timestamps with ground truth
                # Optimized timestamp matching using vectorized operations
                vel_times = vel_ts[:, 0]
                gt_times = gt_ts.unsqueeze(0)  # Shape: [1, N_gt]
                vel_times_expanded = vel_times.unsqueeze(1)  # Shape: [N_vel, 1]
                # Compute all differences at once
                # time_diffs = torch.abs(gt_times.half() - vel_times_expanded.half())  # Shape: [N_vel, N_gt]
                # indices = torch.argmin(time_diffs, dim=1)  # Shape: [N_vel]
                indices = match_nearest_two_pointer(gt_ts, vel_ts[:, 0])
                # === COORDINATE FRAME HANDLING ===
                print("\n🔍 COORDINATE FRAME STATUS:")
                print(f"   World frame: Y-up CPF-aligned (yaw=0 at t=30s)")
                
                print(f"   NO ALIGNMENT NEEDED - all data pre-aligned")
                
                if "coordinate" in dataset_conf.keys() and dataset_conf["coordinate"] == "body_coord":
                    # Get CPF→CPF-world orientations
                    rotation = motion_dataset.data['gt_orientation']
                    
                    # Network outputs are in CPF body frame
                    net_vel_cpf = inference_state['net_vel'].float()
                    
                    # Transform CPF body frame velocities to world frame for evaluation
                    
                    # For velocity error calculation, transform to world (vectorized)
                    # Get rotation matrices for the indices
                    valid_indices = torch.clamp(indices, max=len(rotation)-1)
                    R_cpf_to_world = rotation[valid_indices].matrix().float()  # [N, 3, 3]
                    
                    # Transform network velocities: [N, 3, 3] @ [N, 3, 1] -> [N, 3]
                    net_vel_cpf_subset = net_vel_cpf[:len(valid_indices)]
                    net_vel_world = torch.bmm(R_cpf_to_world[:len(net_vel_cpf_subset)], 
                                             net_vel_cpf_subset.unsqueeze(-1)).squeeze(-1)
                    
                    # Transform GT velocities to world frame
                    gt_vel_cpf = motion_dataset.data['velocity'][indices,:].float()
                    gt_vel_world = torch.bmm(R_cpf_to_world, 
                                            gt_vel_cpf.unsqueeze(-1)).squeeze(-1)
                    
                    vel_dist = net_vel_world - gt_vel_world
                    
                    # Interpolate network velocity to GT timeline
                    gt_ts_np = gt_ts.detach().cpu().numpy()
                    vel_ts_np = vel_ts[:,0].detach().cpu().numpy() 
                    net_vel_np = inference_state['net_vel'].detach().cpu().numpy()
                    
                    net_vel = interp_xyz(gt_ts_np, vel_ts_np, net_vel_np)
                    if isinstance(net_vel, np.ndarray):
                        net_vel = torch.tensor(net_vel, dtype=torch.float32)
                    else:
                        net_vel = net_vel.float()
                    
                    # Interpolated network velocities ready for integration
                
                elif "coordinate" in dataset_conf.keys() and dataset_conf["coordinate"] == "glob_coord":
                    # WRAPPER SPECIAL HANDLING: Mixed frame (X,Z body, Y world)
                    # Transform X,Z to world using timestamp-matched rotations
                    print("\\n🌍 WRAPPER EVALUATION (Hybrid fusion: body→world + wrapper Y):")

                    # Get FULL body velocity and wrapper Y prediction
                    gt_ts_np = gt_ts.detach().cpu().numpy()
                    vel_ts_np = vel_ts[:,0].detach().cpu().numpy()
                    net_vel_body_np = inference_state['net_vel'].detach().cpu().numpy()  # FULL body velocity
                    wrapper_y_np = inference_state['wrapper_y'].detach().cpu().numpy().squeeze()  # Wrapper's Y prediction [N]

                    print(f"\n📊 WRAPPER FULL BODY VELOCITIES (before interpolation):")
                    print(f"   Shape: {net_vel_body_np.shape}")
                    print(f"   First 5 body velocities (X,Y,Z all in body frame):")
                    for i in range(min(5, len(net_vel_body_np))):
                        print(f"     [{i}]: {net_vel_body_np[i]}")

                    print(f"\n📊 WRAPPER Y PREDICTION (world frame, before interpolation):")
                    print(f"   Shape: {wrapper_y_np.shape}")
                    print(f"   First 5 wrapper Y values: {wrapper_y_np[:5]}")

                    # Interpolate FULL body velocity to GT timeline
                    net_vel_body_interp = interp_xyz(gt_ts_np, vel_ts_np, net_vel_body_np)
                    if isinstance(net_vel_body_interp, np.ndarray):
                        net_vel_body_interp = torch.tensor(net_vel_body_interp, dtype=torch.float32)
                    else:
                        net_vel_body_interp = net_vel_body_interp.float()

                    # Interpolate wrapper Y to GT timeline
                    wrapper_y_interp = np.interp(gt_ts_np, vel_ts_np, wrapper_y_np)
                    wrapper_y_interp = torch.tensor(wrapper_y_interp, dtype=torch.float32)

                    print(f"\n📊 AFTER INTERPOLATION:")
                    print(f"   Body velocity shape: {net_vel_body_interp.shape}")
                    print(f"   Wrapper Y shape: {wrapper_y_interp.shape}")
                    print(f"   First 5 interpolated body velocities:")
                    for i in range(min(5, len(net_vel_body_interp))):
                        print(f"     [{i}]: {net_vel_body_interp[i].detach().cpu().numpy()}")

                    # Transform FULL body velocity to world frame
                    print(f"\n  ⚠️  Transforming FULL body velocity to world frame...")
                    gt_orientation_all = motion_dataset.data['gt_orientation']
                    R_all = gt_orientation_all.matrix().float()

                    # Transform using GT orientations at each timestamp
                    net_vel_world = torch.bmm(R_all[:len(net_vel_body_interp)],
                                              net_vel_body_interp.unsqueeze(-1)).squeeze(-1)

                    # Replace Y with wrapper's prediction
                    net_vel = net_vel_world.clone()
                    net_vel[:, 1] = wrapper_y_interp  # Replace Y with wrapper prediction

                    print(f"\n📊 FINAL WORLD VELOCITIES (X,Z from body transform, Y from wrapper):")
                    print(f"   First 5 world velocities:")
                    for i in range(min(5, len(net_vel))):
                        print(f"     [{i}]: {net_vel[i].detach().cpu().numpy()}")
                    print(f"  ✓ Transformed full body velocity, replaced Y with wrapper prediction")

                if data_conf.name == "BlackBird":
                    save_prefix = os.path.dirname(data_name).split('/')[1]
                else:
                    save_prefix = data_name
               
                # Calculate dt for integration
                gt_dt = (gt_ts[1:] - gt_ts[:-1]).mean()
                dt = torch.full((len(net_vel)-1,), gt_dt, dtype=net_vel.dtype, device=net_vel.device)
                gt_positions = motion_dataset.data['gt_translation']
                
                # Prepare data for integration
                if "coordinate" in dataset_conf.keys() and dataset_conf["coordinate"] == "glob_coord":
                    # World frame integration - dataset has pre-transformed velocities to world frame
                    print(f"\\n🌍 Preparing world frame integration:")
                    print(f"   Using world frame velocities (pre-transformed by dataset)")
                    
                    # World frame integration setup complete
                    
                    data_inte = {
                        "vel": net_vel,         # World frame velocity (pre-transformed by dataset)
                        'dt': dt,              # Time steps
                        'coordinate': 'glob_coord',  # Flag for world frame
                        'gt_position': gt_positions 
                    }
                else:
                    # Body-frame integration with orientation transformation
                    data_inte = {
                        "vel": net_vel,        # CPF body frame velocity
                        'dt': dt,              # Time steps
                        'orientation': motion_dataset.data['gt_orientation'],  # CPF→CPF-world orientations
                        'gt_position': gt_positions 
                    }
                
                # Start trajectory at origin for fair comparison
                integrator_vel = Velocity_Integrator(
                    torch.zeros(3, dtype=torch.float64)).to(args.device).double()
                
                # Choose integration method
                if args.use_msckf and MSCKF_AVAILABLE:
                    # Use MSCKF for sliding window integration
                    inf_outstate = integrate_with_msckf(motion_dataset, inference_state, gravity, args)
                    if inf_outstate is None:
                        print("   MSCKF integration failed, falling back to standard integration")
                        # Fallback to standard integration
                        init_relative = init.copy()
                        init_relative['pos'] = torch.zeros_like(init['pos'])
                        inf_outstate = integrate_pos_with_orientation(
                            integrator_vel, data_inte, init_relative, motion_dataset,
                            device=args.device
                        )
                else:
                    if args.use_msckf and not MSCKF_AVAILABLE:
                        print("   Warning: MSCKF requested but not available, using standard integration")
                    
                    # Choose integration method based on coordinate system
                    init_relative = init.copy()
                    init_relative['pos'] = torch.zeros_like(init['pos'])
                    
                    if "coordinate" in dataset_conf.keys() and dataset_conf["coordinate"] == "glob_coord":
                        # World frame integration - velocities pre-transformed by dataset
                        inf_outstate = integrate_pos_world_frame(
                            integrator_vel, data_inte, init_relative, motion_dataset,
                            device=args.device
                        )
                    else:
                        # Body frame integration with orientation transformation
                        inf_outstate = integrate_pos_with_orientation(
                            integrator_vel, data_inte, init_relative, motion_dataset,
                            device=args.device
                        )
                
                # Fix shape mismatch if needed
                if len(inf_outstate['poses'][0]) != len(inf_outstate['poses_gt'][0]):
                    min_len = min(len(inf_outstate['poses'][0]), len(inf_outstate['poses_gt'][0]))
                    inf_outstate['poses'] = inf_outstate['poses'][:, :min_len]
                    inf_outstate['poses_gt'] = inf_outstate['poses_gt'][:, :min_len]
                    print(f"⚠️ Aligned trajectory lengths to {min_len} samples")
                
                # save the translation into pickle file
                pred_trans = inf_outstate['poses'].cpu() 
                gt_trans = inf_outstate['poses_gt'].cpu()
                
                updated_inference_state['pred_trans'] = pred_trans
                updated_inference_state['gt_trans'] = gt_trans
                
                # Calculate metrics
                inf_rte_200 = calculate_rte(inf_outstate, 200, args.seqlen)
                inf_rte_1000 = calculate_rte(inf_outstate, 1000, args.seqlen)
                
                # Position statistics
                print(f"\n📊 Position statistics (Y-up CPF-world, yaw=0 aligned):")
                print(f"   Predicted trajectory shape: {inf_outstate['poses'].shape}")
                print(f"   First position: {inf_outstate['poses'][0, 0, :]}")
                print(f"   Last position: {inf_outstate['poses'][0, -1, :]}")
                # Calculate trajectory length for standard method
                pred_steps = torch.diff(inf_outstate['poses'][0], dim=0)
                pred_trajectory_length = pred_steps.norm(dim=1).sum().item()
                
                print(f"   Total displacement: {(inf_outstate['poses'][0, -1, :] - inf_outstate['poses'][0, 0, :]).norm():.2f}m")
                print(f"   Trajectory length: {pred_trajectory_length:.2f}m")
                
                print(f"\n   GT trajectory shape: {inf_outstate['poses_gt'].shape}")
                print(f"   GT first position: {inf_outstate['poses_gt'][0, 0, :]}")
                print(f"   GT last position: {inf_outstate['poses_gt'][0, -1, :]}")
                
                # Calculate GT trajectory length
                gt_steps = torch.diff(inf_outstate['poses_gt'][0], dim=0)
                gt_trajectory_length = gt_steps.norm(dim=1).sum().item()
                
                print(f"   GT total displacement: {(inf_outstate['poses_gt'][0, -1, :] - inf_outstate['poses_gt'][0, 0, :]).norm():.2f}m")
                print(f"   GT trajectory length: {gt_trajectory_length:.2f}m")
                
                traj_length = (inf_outstate['poses_gt'][0, 1:] - inf_outstate['poses_gt'][0, :-1]).norm(dim=-1).sum().item()
                final_error = inf_outstate['pos_dist'][-1].item()
                drift_rate = (final_error / traj_length * 100) if traj_length > 0 else 0.0

                # Save loss result
                result_dic = {
                    'name': data_name,      
                    'ATE': torch.sqrt((inf_outstate['pos_dist']**2).mean()).item(),
                    'ATE_x': torch.sqrt((inf_outstate['pos_dist_x']**2).mean()).item(),
                    'ATE_y': torch.sqrt((inf_outstate['pos_dist_y']**2).mean()).item(),
                    'ATE_z': torch.sqrt((inf_outstate['pos_dist_z']**2).mean()).item(),
                    'ATE_xy': torch.sqrt((inf_outstate['pos_dist_xy']**2).mean()).item(),
                    'ATE_yz': torch.sqrt((inf_outstate['pos_dist_yz']**2).mean()).item(),
                    'ATE_xz': torch.sqrt((inf_outstate['pos_dist_xz']**2).mean()).item(),
                    'AVE': inf_outstate['vel_dist'].mean().item(),
                    'RP_RMSE_200': np.sqrt((inf_rte_200**2).mean()).numpy().item() if len(inf_rte_200) > 0 else None,
                    'RP_RMSE_1000': np.sqrt((inf_rte_1000**2).mean()).numpy().item() if len(inf_rte_1000) > 0 else None,
                    'drift_rate': drift_rate,  # ADD THIS LINE
                    'method': method_suffix,
                    'final_pos_error': inf_outstate['pos_dist'][-1].item(),
                    'trajectory_length': len(inf_outstate['poses'][0])
                }
                
                AllResults.append(result_dic)
                

            # === VISUALIZATION ===
            if args.vis:
                # Visualize trajectories in Y-up coordinate system (pre-aligned)
                print("\n🎨 Creating visualization (Y-up CPF-world, pre-aligned)...")
                
                # No rotation needed - everything is pre-aligned
                rotation_angle = 0
                scale_factor = 1.0
                
                # Create 3D visualization (Y-up world)
                visualize_3D_motion_with_xz(save_prefix, folder, outstate, inf_outstate if args.exp else None, 
                                   label="AirIO_All")

        with open(os.path.join(args.exp, 'net_output_with_trans.pickle'), 'wb') as f:
            pickle.dump(updated_inference_state, f)

        # After the loop through all data_drive_list
        if len(AllResults) > 0:
            # Add summary statistics
            summary = {
                'total_sequences': len(AllResults),
                'avg_ATE': np.mean([r['ATE'] for r in AllResults]),
                'std_ATE': np.std([r['ATE'] for r in AllResults]),
                'avg_ATE_x': np.mean([r['ATE_x'] for r in AllResults]),
                'std_ATE_x': np.std([r['ATE_x'] for r in AllResults]),
                'avg_ATE_y': np.mean([r['ATE_y'] for r in AllResults]),
                'std_ATE_y': np.std([r['ATE_y'] for r in AllResults]),
                'avg_ATE_z': np.mean([r['ATE_z'] for r in AllResults]),
                'std_ATE_z': np.std([r['ATE_z'] for r in AllResults]),
                'avg_ATE_xy': np.mean([r['ATE_xy'] for r in AllResults]),
                'std_ATE_xy': np.std([r['ATE_xy'] for r in AllResults]),
                'avg_ATE_yz': np.mean([r['ATE_yz'] for r in AllResults]),
                'std_ATE_yz': np.std([r['ATE_yz'] for r in AllResults]),
                'avg_ATE_xz': np.mean([r['ATE_xz'] for r in AllResults]),
                'std_ATE_xz': np.std([r['ATE_xz'] for r in AllResults]),
                'avg_RTE_200': np.mean([r['RP_RMSE_200'] for r in AllResults if r["RP_RMSE_200"] is not None]),
                'std_RTE_200': np.std([r['RP_RMSE_200'] for r in AllResults if r["RP_RMSE_200"] is not None]),
                'avg_RTE_1000': np.mean([r['RP_RMSE_1000'] for r in AllResults if r["RP_RMSE_1000"] is not None]),
                'std_RTE_1000': np.std([r['RP_RMSE_1000'] for r in AllResults if r["RP_RMSE_1000"] is not None]),
                'avg_drift_rate': np.mean([r['drift_rate'] for r in AllResults]),
                'std_drift_rate': np.std([r['drift_rate'] for r in AllResults]),
                'avg_AVE': np.mean([r['AVE'] for r in AllResults]),
                'std_AVE': np.std([r['AVE'] for r in AllResults])
            }

            print("\n" + "="*70)
            print(f"AGGREGATE RESULTS ({method_suffix.upper()} METHOD)")
            print("="*70)
            print(f"Configuration: {config_name}")
            print(f"Integration method: {method_suffix}")
            if args.use_msckf:
                print(f"MSCKF sliding window states: Enabled")
            print(f"Total sequences evaluated: {summary['total_sequences']}")
            print(f"ATE: {summary['avg_ATE']:.4f} ± {summary['std_ATE']:.4f} m")
            print(f"ATE_x: {summary['avg_ATE_x']:.4f} ± {summary['std_ATE_x']:.4f} m")
            print(f"ATE_y: {summary['avg_ATE_y']:.4f} ± {summary['std_ATE_y']:.4f} m")
            print(f"ATE_z: {summary['avg_ATE_z']:.4f} ± {summary['std_ATE_z']:.4f} m")
            print(f"ATE_xy: {summary['avg_ATE_xy']:.4f} ± {summary['std_ATE_xy']:.4f} m")
            print(f"ATE_yz: {summary['avg_ATE_yz']:.4f} ± {summary['std_ATE_yz']:.4f} m")       
            print(f"ATE_xz: {summary['avg_ATE_xz']:.4f} ± {summary['std_ATE_xz']:.4f} m")
            print(f"RTE_200: {summary['avg_RTE_200']:.4f} ± {summary['std_RTE_200']:.4f} m")
            print(f"RTE_1000: {summary['avg_RTE_1000']:.4f} ± {summary['std_RTE_1000']:.4f} m")
            print(f"Drift: {summary['avg_drift_rate']:.2f} ± {summary['std_drift_rate']:.2f} %")
            print(f"AVE: {summary['avg_AVE']:.4f} ± {summary['std_AVE']:.4f} m/s")
            
            # Save summary with results
            AllResults.append({'summary': summary})
            
        # Save all results
        file_path = os.path.join(folder, "result.json")
        with open(file_path, 'w') as f: 
            json.dump(AllResults, f, indent=4)
        
        print(f"\n✅ Results saved to: {file_path}")
        print("\n" + "="*70)
        print("EVALUATION COMPLETE")
        print("="*70)