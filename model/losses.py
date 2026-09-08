import torch
import torch.nn.functional as F
import pypose as pp
from .loss_func import loss_fc_list, diag_ln_cov_loss

def motion_loss_(fc, pred, targ):
    dist = pred - targ
    loss = fc(dist)
    return loss, dist


def resample_time(x: torch.Tensor, T_out: int) -> torch.Tensor:
    # x: (B, T, 3) -> (B, 3, T)
    x_ch_first = x.permute(0, 2, 1).contiguous()
    # 1D linear interpolation along time to size T_out
    x_resampled = F.interpolate(x_ch_first, size=T_out, mode='linear', align_corners=False)
    # back to (B, T, 3)
    return x_resampled.permute(0, 2, 1).contiguous()


def get_motion_loss(inte_state, label, confs, data=None, gravity_world=None):
    """
    Enhanced loss function with proper quaternion/log handling
    
    Args:
        gravity_world: World gravity vector, default [0, 0, -9.81]
    """
    loss = 0
    loss_fc = loss_fc_list[confs.loss]
    l2_fc = loss_fc_list["L2"]
    
    vel_pred = inte_state['net_vel']
    vel_gt = label
    
    # Standard velocity matching
    vel_loss, vel_dist = motion_loss_(loss_fc, vel_pred, vel_gt)
    l2_vel_dist, _ = motion_loss_(l2_fc, vel_pred, vel_gt)
    
    # Check for sparsity weight in config
    sparsity_weight = getattr(confs, 'sparsity_weight', 0.0)
    if sparsity_weight > 0:
        # Add L1 penalty to encourage exact zeros when stationary
        l1_penalty = torch.abs(vel_pred).mean()
        vel_loss += sparsity_weight * l1_penalty
    
    # Check for adaptive threshold weight
    adaptive_weight = getattr(confs, 'adaptive_threshold_weight', 0.0)
    if adaptive_weight > 0:
        vel_gt_mag = torch.norm(vel_gt, dim=-1, keepdim=True)
        threshold = getattr(confs, 'vel_threshold', 0.1)
        is_stationary = (vel_gt_mag < threshold).float()
        weight = 1.0 + is_stationary * 2.0  # 3x weight when stationary
        vel_error = (vel_pred - vel_gt).pow(2)
        adaptive_loss = (vel_error * weight).mean()
        vel_loss += adaptive_weight * adaptive_loss
        
    # Temporal smoothness
    if hasattr(confs, 'temporal_smooth_weight') and confs.temporal_smooth_weight > 0:
        if vel_pred.shape[1] > 1:
            vel_diff_pred = vel_pred[:, 1:] - vel_pred[:, :-1]
            vel_diff_target = vel_gt[:, 1:] - vel_gt[:, :-1]
            temporal_loss = loss_fc(vel_diff_pred - vel_diff_target)
            vel_loss += confs.temporal_smooth_weight * temporal_loss

    # Position-based losses

    vel_pred_body = inte_state['net_vel']
    vel_gt_body = label
    rot = data['rot']
    
    # Get dt
    dt = data['dt'][:, :-1] if 'dt' in data else 0.02
    if not isinstance(dt, torch.Tensor):
        dt = torch.tensor(dt, device=vel_pred_body.device)
    if dt.dim() < 3:
        if dt.dim() == 0:
            dt = dt.expand(vel_pred_body.shape[0], vel_pred_body.shape[1], 1)
        elif dt.dim() == 1:
            dt = dt.unsqueeze(0).unsqueeze(-1)
        elif dt.dim() == 2:
            dt = dt.unsqueeze(-1)
    # Check if we're already in world frame (glob_coord)
    # If data is in world frame, velocities are already in world frame
    if hasattr(confs, 'coordinate') == False:
        raise ValueError("`coordinate` must be set in `confs` (e.g., 'glob_coord' or 'body_coord').")
    
    if getattr(confs, 'coordinate', None) == 'glob_coord':
        # Velocities are already in world frame, no transformation needed
        vel_pred_world = vel_pred_body  # Already world frame
        vel_gt_world = vel_gt_body      # Already world frame

        # Downsample dt to match velocity temporal dimension if needed
        # dt is (B, T_dt, 1) after reshaping above
        if dt.size(1) != vel_pred_world.size(1):
            # dt is (B, T_dt, 1), velocity is (B, T_vel, 3), need to downsample dt
            # resample_time expects (B, T, C) format
            dt = resample_time(dt, vel_pred_world.size(1))  # (B, T_vel, 1)
    else:
        # Transform from body to world frame using rotations
        # === FIXED: Handle different rotation representations ===
        if hasattr(rot, 'matrix'):
            # It's already a PyPose LieTensor
            R = rot.matrix()
        elif rot.shape[-1] == 12:
            # Triple quaternion (12D): [rot_R(4), rot_L(4), rot_rel(4)]
            # For position loss, use the first quaternion (right IMU/CPF frame)
            rot_single = rot[..., :4]
            R = pp.SO3(rot_single).matrix()
        elif rot.shape[-1] == 8:
            # Dual quaternion (8D): [rot_R(4), rot_L(4)]
            # For position loss, use the first quaternion (right IMU) or average
            # Using first quaternion for simplicity
            rot_single = rot[..., :4]
            R = pp.SO3(rot_single).matrix()
        elif rot.shape[-1] == 4:
            # It's a quaternion tensor [qx, qy, qz, qw]
            # Convert to SO3 then to matrix
            R = pp.SO3(rot).matrix()
        elif rot.shape[-1] == 3:
            # It's a log representation
            R = pp.so3(rot).Exp().matrix()
        else:
            raise ValueError(f"Unexpected rotation shape: {rot.shape}")
        
        # Transform velocities to world frame
        if R.size(1) != vel_pred_body.size(1):
            T_target = R.size(1)  # 1000
            vel_pred_body = resample_time(vel_pred_body, T_target)  # (B, 1000, 3)
            vel_gt_body   = resample_time(vel_gt_body,   T_target)  # (B, 1000, 3)

        vel_pred_world = torch.einsum('btij,btj->bti', R, vel_pred_body)
        vel_gt_world = torch.einsum('btij,btj->bti', R, vel_gt_body)
    
    if confs.position_weight != 0 and confs.final_position_weight != 0:
        # Integrate using cumsum
        displacements_pred = vel_pred_world * dt
        displacements_gt = vel_gt_world * dt
        
        pos_pred = torch.cumsum(displacements_pred, dim=1)
        pos_gt = torch.cumsum(displacements_gt, dim=1)
        
        # Position matching loss
        pos_error = (pos_pred - pos_gt).pow(2)
        position_loss = pos_error.mean()
        loss += confs.position_weight * position_loss

        # Final position loss
        final_pos_pred = pos_pred[:, -1, :]
        final_pos_gt = pos_gt[:, -1, :]
        final_pos_error = (final_pos_pred - final_pos_gt).pow(2)
        final_position_loss = final_pos_error.mean()
        loss += confs.final_position_weight * final_position_loss
    else:
        position_loss = torch.tensor(0.0, device=vel_pred.device)
        final_position_loss = torch.tensor(0.0, device=vel_pred.device)
    
    # Covariance loss
    cov_loss = 0
    if confs.propcov:
        cov = inte_state['cov']
        cov_loss = cov.mean()
        if "covaug" in confs and confs["covaug"]:
            vel_loss += confs.cov_weight * diag_ln_cov_loss(vel_dist, cov)
        else:
            vel_loss += confs.cov_weight * diag_ln_cov_loss(vel_dist.detach(), cov)
    
    loss += confs.weight * vel_loss

    return {'loss': loss, 'cov_loss': cov_loss, 'vel_l2': l2_vel_dist, 'position_loss': position_loss, "final_position_loss": final_position_loss}

def get_motion_RMSE(inte_state, label, confs):
    '''
    get the RMSE of the last state in one segment
    '''
    def _RMSE(x):
        return torch.sqrt((x.norm(dim=-1)**2).mean())
    
    cov_loss = 0
    dist = (inte_state['net_vel'] - label)
    
    # Handle both downsampled and full-resolution models
    if dist.shape[1] > 1:
        dist = torch.mean(dist, dim=-2)
    
    loss = _RMSE(dist)[None,...]
    
    if confs.propcov:
        cov = inte_state['cov']
        cov_loss = cov.mean()
    
    return {'loss': loss,
            'dist': dist.norm(dim=-1).mean(),
            'cov_loss': cov_loss}
    
def get_joint_loss(inte_state, gt_joint_pos, gt_joint_rot, confs):
    loss = 0
    
    l2_fc = loss_fc_list["L2"]
    
    joint_pos_pred = inte_state["joint_pos"]
    joint_rot_pred = inte_state["joint_rot"]
    
    joint_pos_loss, _ = motion_loss_(l2_fc, joint_pos_pred, gt_joint_pos)
    joint_rot_loss, _ = motion_loss_(l2_fc, joint_rot_pred, gt_joint_rot)
    
    joint_pos_loss = confs.joint_pos_weight * joint_pos_loss
    joint_rot_loss = confs.joint_rot_weight * joint_rot_loss
    
    loss = confs.joint_pos_weight * joint_pos_loss + confs.joint_rot_weight * joint_rot_loss
    
    return {
        "loss": loss,
        "joint_pos_loss": joint_pos_loss , 
        "joint_rot_loss": joint_rot_loss
    }
    
    
    