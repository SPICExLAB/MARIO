import torch
from torch import nn

from utils import move_to


class Velocity_Integrator(nn.Module):
    def __init__(self, pos = torch.zeros(3)):
        super().__init__()
        self.register_buffer('pos',self._check(pos).clone(), persistent=False)
        
    def _check(self, obj):
        if obj is not None:
            if len(obj.shape) == 2:
                obj = obj[None, ...]

            elif len(obj.shape) == 1:
                obj = obj[None, None, ...]
        return obj

    def forward(self, dt, vel, init_state=None):
        dt = self._check(dt)
        B = dt.shape[0]

        if init_state is None:
            init_state = {'pos': self.pos}
          
        predict = self.integrate(dt,vel,init_state)
        
        self.pos = predict['pos'][...,-1,:]
        
        return {**predict}
    
    def integrate(self, dt, vel,init_state):
        B, F = dt.shape[:2]

        dp = torch.zeros(B,1,3,dtype = dt.dtype, device = dt.device)
        # dp = torch.cat([dp, 0.5 * (vel[:,:F]+vel[:,1:])*dt ],dim=1)

        dp = torch.cat([dp, vel[:,:F]*dt],dim=1)
        incre_p = torch.cumsum(dp,dim=1)

        # Ensure init_state['pos'] is on the same device as incre_p
        init_pos = init_state['pos'].to(device=incre_p.device, dtype=incre_p.dtype)

        return {'pos': init_pos + incre_p[:,1:,:]}
   
 
# #################TEST#############
def integrate_pos(integrator, datainte, init,dataset, device="cpu"):
    out_state = dict()
    vel_gt, poses_gt = [init['vel'][None,:]],[init['pos'][None,:]]
    state = integrator(
            dt=datainte['dt'][...,None],vel=datainte["vel"][None,...]
        )

    # Ensure all tensors are on the same device
    out_state['poses'] = state["pos"].to(device=device)
    out_state['net_vel'] = datainte["vel"][None,...].to(device=device)
    out_state['vel_gt'] = dataset.data['velocity'].to(device=device)
    out_state['poses_gt'] = dataset.data['gt_translation'].to(device=device)
    out_state['pos_dist'] = (out_state['poses'] - out_state['poses_gt'][1:out_state['poses'].shape[0]+1,:]).norm(dim=-1)
    out_state['vel_dist'] = (datainte["vel"].to(device=device) -  out_state['vel_gt']).norm(dim=-1)
    out_state['vel_mag_dist'] = torch.abs(datainte["vel"].to(device=device).norm(dim=-1) - out_state['vel_gt'].norm(dim=-1))
    out_state['vel_error'] = (datainte["vel"].to(device=device) -  out_state['vel_gt'])

    # Optional: joint_rot only exists in skeleton datasets, not IMU-only datasets like TLIO
    if 'joint_rot' in dataset.data:
        out_state['joint_rot_gt'] = dataset.data['joint_rot'].to(device=device)

    return out_state

def integrate_pos_world_frame(integrator, datainte, init, dataset, device="cpu"):
    """
    Integrate world-frame velocities directly (no orientation transformation needed)
    
    Args:
        integrator: Velocity integrator
        datainte: Dict with 'vel' (world frame), 'dt', 'coordinate' flag
        init: Initial state
        dataset: Dataset for GT comparison
        device: Device
    
    Returns:
        out_state: Dict with poses in world frame
    """
    out_state = dict()
    
    world_vel = datainte["vel"]  # World frame velocity [N, 3]
    dt = datainte['dt']         # Time steps [N-1]
    
    print(f"🌍 Integrating {len(world_vel)} world-frame velocity samples...")
    print(f"   World velocity shape: {world_vel.shape}")
    print(f"   Initial position: {init['pos']}")
    print(f"   World velocity magnitude range: [{world_vel.norm(dim=-1).min():.4f}, {world_vel.norm(dim=-1).max():.4f}]")
    
    # Direct integration in world frame
    state = integrator(
        dt=dt[...,None],
        vel=world_vel[None,...]  # Add batch dimension
    )
    
    # GT comparison - use filtered GT velocity if provided, otherwise use full dataset
    if 'gt_velocity' in datainte:
        gt_world_velocities = datainte['gt_velocity']  # Filtered GT velocity matching network timeline
    else:
        gt_world_velocities = dataset.data['velocity']  # GT velocities already in world frame
    target_device = gt_world_velocities.device

    # Move all tensors to target device (cuda:0)
    out_state['poses'] = state["pos"].to(device=target_device)  # Already in world frame
    out_state['net_vel'] = world_vel[None,...].to(device=target_device)  # Network predictions in world frame
    out_state['vel_gt'] = gt_world_velocities
    out_state["poses_gt"] = datainte['gt_position'][None,...]

    # Calculate errors
    min_len = min(out_state['poses'].shape[1], out_state['poses_gt'].shape[1])
    out_state['poses'] = out_state['poses'][:, :min_len]
    out_state['poses_gt'] = out_state['poses_gt'][:, :min_len]
    
    # Ensure world_vel is on same device for error calculations
    world_vel_device = world_vel.to(device=target_device)

    out_state['pos_dist'] = (out_state['poses'][0] - out_state['poses_gt'][0]).norm(dim=-1)
    out_state['pos_dist_x'] = (out_state['poses'][0,:,0] - out_state['poses_gt'][0,:,0]).abs()
    out_state['pos_dist_y'] = (out_state['poses'][0,:,1] - out_state['poses_gt'][0,:,1]).abs()
    out_state['pos_dist_z'] = (out_state['poses'][0,:,2] - out_state['poses_gt'][0,:,2]).abs()

    out_state['pos_dist_xy'] = (out_state['poses'][0,:, :2] - out_state['poses_gt'][0,:,:2]).norm(dim=-1)
    out_state['pos_dist_xz'] = (out_state['poses'][0,:, [0, 2]] - out_state['poses_gt'][0,:,[0,2]]).norm(dim=-1)
    out_state['pos_dist_yz'] = (out_state['poses'][0,:, 1:] -  out_state['poses_gt'][0,:,1:]).norm(dim=-1) 
    
    out_state['vel_dist'] = (world_vel[:len(gt_world_velocities)] - gt_world_velocities[:len(world_vel)]).norm(dim=-1)
    out_state['vel_mag_dist'] = torch.abs(world_vel[:len(gt_world_velocities)].norm(dim=-1) - gt_world_velocities[:len(world_vel)].norm(dim=-1))
    out_state['vel_error'] = (world_vel[:len(gt_world_velocities)] - gt_world_velocities[:len(world_vel)])
    
    print(f"✅ World frame integration complete:")
    print(f"   Final position: {out_state['poses'][0, -1]}")
    print(f"   Final position error: {out_state['pos_dist'][-1]:.4f}m")
    print(f"   Mean velocity error: {out_state['vel_dist'].mean():.4f}m/s")
    
    return out_state

def integrate_pos_with_orientation(integrator, datainte, init, dataset, device="cpu"):
    """
    Integrate body-frame velocities using orientation to get world-frame positions (VECTORIZED)
    
    Args:
        integrator: Velocity integrator (unused, kept for compatibility)
        datainte: Dict with 'vel' (body frame), 'dt', 'orientation' (world→body)
        init: Initial state
        dataset: Dataset for GT comparison
        device: Device
    
    Returns:
        out_state: Dict with poses in world frame
    """
    out_state = dict()
    
    body_vel = datainte["vel"]  # CPF/body frame velocity [N, 3]
    dt = datainte['dt']         # Time steps [N-1]
    orientation = datainte['orientation']  # World→CPF quaternions [N, 4]
    
    print(f"🔄 Integrating {len(body_vel)} body-frame velocity samples (VECTORIZED)...")
    print(f"   Body velocity shape: {body_vel.shape}")
    print(f"   Orientation shape: {orientation.shape}")
    print(f"   Initial position: {init['pos']}")
    
    # DEBUG: Check actual values
    print(f"\n📊 BODY VELOCITY ANALYSIS:")
    print(f"   First 5 body velocities:")
    for i in range(min(5, len(body_vel))):
        print(f"     [{i}]: {body_vel[i].detach().cpu().numpy()}")
    print(f"   Body velocity magnitude stats: min={body_vel.norm(dim=-1).min():.4f}, max={body_vel.norm(dim=-1).max():.4f}, mean={body_vel.norm(dim=-1).mean():.4f}")
    
    print(f"\n📊 ORIENTATION ANALYSIS:")
    print(f"   First 3 orientations (CPF→world quaternions):")
    for i in range(min(3, len(orientation))):
        print(f"     [{i}]: {orientation[i].tensor().detach().cpu().numpy()}")
    
    # VECTORIZED: Transform all body velocities to world frame at once
    # orientation gives CPF→World transformations (no .Inv() needed!)
    
    # DEBUG: Check transformation magnitudes
    print(f"\n🔬 COORDINATE TRANSFORMATION DEBUG:")
    # Match the dtype and device of the orientation tensor
    target_dtype = orientation.dtype
    target_device = orientation.device
    sample_cpf_vel = body_vel[0].to(dtype=target_dtype, device=target_device)  # Match orientation dtype and device
    sample_orientation = orientation[0]
    sample_world_vel = sample_orientation @ sample_cpf_vel
    
    print(f"   CPF velocity [0]: {sample_cpf_vel.detach().cpu().numpy()}")
    print(f"   Orientation [0]: {sample_orientation.tensor().detach().cpu().numpy()}")
    print(f"   World velocity [0]: {sample_world_vel.detach().cpu().numpy()}")
    print(f"   CPF magnitude: {sample_cpf_vel.norm():.6f}")
    print(f"   World magnitude: {sample_world_vel.norm():.6f}")
    print(f"   Magnitude ratio (world/cpf): {(sample_world_vel.norm() / sample_cpf_vel.norm()).item():.3f}")
    print(f"   Rotation preserves magnitude: {torch.isclose(sample_cpf_vel.norm(), sample_world_vel.norm(), atol=1e-6).item()}")
    
    # Match dtype and device for all tensor operations
    world_velocities = orientation @ body_vel.to(dtype=target_dtype, device=target_device)  # [N, 3] - CPF→world transformation, no .Inv() needed!

    # ALSO transform GT velocities to world frame for fair comparison
    # Use filtered GT velocities if provided in datainte, otherwise use full dataset
    if 'gt_velocity' in datainte:
        gt_velocities_cpf = datainte['gt_velocity'].to(dtype=target_dtype, device=target_device)
    else:
        gt_velocities_cpf = dataset.data['velocity'].to(dtype=target_dtype, device=target_device)  # GT velocities in CPF frame - Match dtype and device
    gt_world_velocities = orientation @ gt_velocities_cpf  # Transform to world frame
    
    # Skip position differential validation - not needed for production
    
    print(f"\n📊 WORLD VELOCITY ANALYSIS:")
    print(f"   First 5 world velocities:")
    for i in range(min(5, len(world_velocities))):
        print(f"     [{i}]: {world_velocities[i].detach().cpu().numpy()}")
    print(f"   World velocity magnitude stats: min={world_velocities.norm(dim=-1).min():.4f}, max={world_velocities.norm(dim=-1).max():.4f}, mean={world_velocities.norm(dim=-1).mean():.4f}")
    
    print(f"\n📊 GT WORLD VELOCITY ANALYSIS:")
    print(f"   First 5 GT world velocities:")
    for i in range(min(5, len(gt_world_velocities))):
        print(f"     [{i}]: {gt_world_velocities[i].detach().cpu().numpy()}")
    print(f"   GT world velocity magnitude stats: min={gt_world_velocities.norm(dim=-1).min():.4f}, max={gt_world_velocities.norm(dim=-1).max():.4f}, mean={gt_world_velocities.norm(dim=-1).mean():.4f}")
    
    # VECTORIZED: Apply dt to velocities for integration
    # dt has shape [N-1], so we need to trim world_velocities
    dt = dt.to(device=target_device)  # Ensure dt is on the same device
    vel_with_dt = world_velocities[:-1] * dt[..., None]  # [N-1, 3]
    
    print(f"\n📊 INTEGRATION ANALYSIS:")
    print(f"   dt shape: {dt.shape}, first 5 dt values: {dt[:5].detach().cpu().numpy()}")
    print(f"   vel_with_dt shape: {vel_with_dt.shape}")
    print(f"   First 3 velocity*dt displacements:")
    for i in range(min(3, len(vel_with_dt))):
        print(f"     [{i}]: vel={world_velocities[i].detach().cpu().numpy()} * dt={dt[i]:.4f} = {vel_with_dt[i].detach().cpu().numpy()}")
    
    # VECTORIZED: Cumulative sum for position integration
    displacements = torch.cumsum(vel_with_dt, dim=0)  # [N-1, 3]
    
    print(f"   First 3 cumulative displacements:")
    for i in range(min(3, len(displacements))):
        print(f"     [{i}]: {displacements[i].detach().cpu().numpy()}")
    
    # Add initial position
    positions = init['pos'].to(device=target_device) + displacements  # [N-1, 3]

    print(f"   First 3 absolute positions:")
    for i in range(min(3, len(positions))):
        print(f"     [{i}]: init_pos + displacement = {init['pos'].detach().cpu().numpy()} + {displacements[i].detach().cpu().numpy()} = {positions[i].detach().cpu().numpy()}")

    # Store number of predicted positions
    num_predicted = positions.shape[0]

    # Store results - keep dtype consistency, but ensure positions are float for PyTorch operations
    out_state['poses'] = positions[None, ...].float().to(device=target_device)  # Add batch dimension [1, N-1, 3] - positions need float for math ops
    out_state['net_vel'] = body_vel[None, ...]  # [1, N, 3] - keep original body_vel dtype
    out_state['vel_gt'] = gt_world_velocities  # Use world-transformed GT velocities - keep target dtype

    out_state['poses_gt'] = datainte['gt_position'][None,...]

    # Predicted trajectory should also start from origin (already does from integration)
    # No adjustment needed - both trajectories now start at origin for relative comparison
    
    # DEBUG: Compare predicted vs GT velocities (both now in world frame)
    print(f"\n📊 VELOCITY COMPARISON (WORLD FRAME):")
    print(f"   GT velocity shape: {out_state['vel_gt'].shape}")
    print(f"   First 5 GT world velocities (transformed from CPF):")
    for i in range(min(5, len(out_state['vel_gt']))):
        print(f"     [{i}]: {out_state['vel_gt'][i].detach().cpu().numpy()}")
    print(f"   GT velocity magnitude stats: min={out_state['vel_gt'].norm(dim=-1).min():.4f}, max={out_state['vel_gt'].norm(dim=-1).max():.4f}, mean={out_state['vel_gt'].norm(dim=-1).mean():.4f}")
    
    print(f"\n📊 VELOCITY DIFFERENCE ANALYSIS (WORLD FRAME):")
    vel_diff = world_velocities - gt_world_velocities
    print(f"   First 5 velocity differences (predicted - GT, both in world frame):") 
    for i in range(min(5, len(vel_diff))):
        print(f"     [{i}]: pred={world_velocities[i].detach().cpu().numpy()} - gt={gt_world_velocities[i].detach().cpu().numpy()} = {vel_diff[i].detach().cpu().numpy()}")
    print(f"   Velocity difference magnitude stats: min={vel_diff.norm(dim=-1).min():.4f}, max={vel_diff.norm(dim=-1).max():.4f}, mean={vel_diff.norm(dim=-1).mean():.4f}")
    
    # Calculate position errors - align shapes properly
    # Handle size mismatch: predicted has N-1 samples, GT has N samples
    num_predicted = out_state['poses'].shape[1]  # N-1
    # GT already trimmed to match in out_state['poses_gt']
    gt_poses_for_comparison = out_state['poses_gt'][0, :num_predicted, :]  # [N-1, 3]
    
    out_state['pos_dist'] = (out_state['poses'][0] - gt_poses_for_comparison).norm(dim=-1)
    out_state['pos_dist_x'] = (out_state['poses'][0,:,0] - gt_poses_for_comparison[:,0]).abs()
    out_state['pos_dist_y'] = (out_state['poses'][0,:,1] - gt_poses_for_comparison[:,1]).abs()
    out_state['pos_dist_z'] = (out_state['poses'][0,:,2] - gt_poses_for_comparison[:,2]).abs()
    
    out_state['pos_dist_xy'] = (out_state['poses'][0,:, :2] - gt_poses_for_comparison[:,:2]).norm(dim=-1)
    out_state['pos_dist_xz'] = (out_state['poses'][0,:, [0, 2]] - gt_poses_for_comparison[:,[0,2]]).norm(dim=-1)
    out_state['pos_dist_yz'] = (out_state['poses'][0,:, 1:] -  gt_poses_for_comparison[:,1:]).norm(dim=-1)   
    
    # Calculate velocity errors - both velocities now in world frame for fair comparison
    # Keep in whatever dtype the orientation tensor was using
    out_state['vel_dist'] = (world_velocities - gt_world_velocities).norm(dim=-1)
    out_state['vel_mag_dist'] = torch.abs(world_velocities.norm(dim=-1) - gt_world_velocities.norm(dim=-1))
    out_state['vel_error'] = (world_velocities - gt_world_velocities)
    
    final_pos = positions[-1]
    total_displacement = (final_pos - init['pos'].to(device=target_device)).norm()
    print(f"✅ Integration complete:")
    print(f"   Final position: {final_pos}")
    print(f"   Total displacement: {total_displacement:.2f}m")
    
    # out_state['joint_rot_gt'] = dataset.data['joint_rot']
    
    return out_state  


if __name__ == "__main__":
    import argparse
    import os

    import torch
    import torch.utils.data as Data
    import tqdm
    from pyhocon import ConfigFactory

    from datasets import SeqDataset, imu_seq_collate
    from utils import CPU_Unpickler, move_to

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device", type=str, default="cpu", help="cuda or cpu, Default is cuda:0"
    )
    parser.add_argument("--exp", type=str, default=None, help="experiment name")
    parser.add_argument(
        "--seqlen", type=int, default="200", help="the length of the segment"
    )
    parser.add_argument(
        "--dataconf",
        type=str,
        default="configs/datasets/SubTDataset/SubT_UGV1_final_half.conf",
        help="the configuration of the dataset",
    )

    args = parser.parse_args()
    print(("\n" * 3) + str(args) + ("\n" * 3))
    config = ConfigFactory.parse_file(args.dataconf)
    dataset_conf = config.inference

    net_result_path = os.path.join(args.exp, "net_output.pickle")
    if os.path.isfile(net_result_path):
        with open(net_result_path, "rb") as handle:
            inference_state_load = CPU_Unpickler(handle).load()
        for data_conf in dataset_conf.data_list:
            for data_name in data_conf.data_drive:
                dataset = SeqDataset(
                    data_conf.data_root,
                    data_name,
                    args.device,
                    name=data_conf.name,
                    duration=args.seqlen,
                    step_size=args.seqlen,
                    drop_last=False,
                    conf=dataset_conf,
                )
                loader = Data.DataLoader(
                    dataset=dataset,
                    batch_size=1,
                    collate_fn=imu_seq_collate,
                    shuffle=False,
                    drop_last=False,
                )
                init = dataset.get_init_value()

                inference_state = inference_state_load[data_name]

                integrator = Velocity_Integrator(
                                    init['pos']).to(args.device).double()
                                
                outstate =integrate_pos(
                                    integrator, data_inte, init, loader,
                                    device=args.device)
                relative_outstate = calculate_rte(outstate, args.seqlen,args.seqlen)
                    
                print("==============Integration==============")
                print("outstate:")
                print("pos_err: ", outstate['pos_dist'].mean())
                print("vel_err: ", outstate['vel_dist'].mean())
                    
                print("relative_state:")
                print("pos_err: ", relative_outstate['pos_dist'].mean())
                   
                    
                    