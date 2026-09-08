"""
Preprocessing utilities for dual-IMU ablation study.

Handles different dual-IMU fusion modes:
- right_only: Use only right IMU (baseline)
- average: Average acc/gyro from both IMUs
- rot_slerp: Average acc/gyro + SLERP rotation
- concat: Concatenate both IMUs
- avg_plus_diff: Average + difference features
- variance_weighted: Optimal MMSE fusion using inverse variance weighting
- variance_tilt: Variance-weighted fusion + tilt correction (Kalman gain)
- dual_accel_tilt: Variance-weighted + dual-accel tilt correction
"""

import torch
import torch.nn.functional as F
import pypose as pp
import numpy as np


def quaternion_inverse(q):
    """
    Compute inverse of quaternion [qx, qy, qz, qw].
    For unit quaternions, inverse = conjugate.

    Args:
        q: (B, T, 4) quaternion tensor

    Returns:
        q_inv: (B, T, 4) inverse quaternion
    """
    # Conjugate: negate xyz, keep w
    q_conj = torch.cat([-q[..., :3], q[..., 3:4]], dim=-1)

    # For unit quaternions, inverse = conjugate / norm^2
    # Since we normalize, we can just return conjugate
    norm_sq = torch.sum(q ** 2, dim=-1, keepdim=True)
    return q_conj / (norm_sq + 1e-8)


def quaternion_multiply(q1, q2):
    """
    Multiply two quaternions q1 * q2 [qx, qy, qz, qw].

    Args:
        q1, q2: (B, T, 4) quaternion tensors

    Returns:
        result: (B, T, 4) product quaternion
    """
    x1, y1, z1, w1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    x2, y2, z2, w2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2

    return torch.stack([x, y, z, w], dim=-1)


def quaternion_slerp(q1, q2, t=0.5):
    """
    Spherical linear interpolation between two quaternions.

    Args:
        q1, q2: (B, T, 4) quaternion tensors
        t: interpolation parameter (0=q1, 1=q2), default 0.5 for average

    Returns:
        q_interp: (B, T, 4) interpolated quaternion
    """
    # Normalize inputs
    q1 = F.normalize(q1, p=2, dim=-1)
    q2 = F.normalize(q2, p=2, dim=-1)

    # Compute dot product
    dot = torch.sum(q1 * q2, dim=-1, keepdim=True)

    # If dot < 0, negate q2 to take shorter path
    q2 = torch.where(dot < 0, -q2, q2)
    dot = torch.abs(dot)

    # If quaternions are very close, use linear interpolation
    DOT_THRESHOLD = 0.9995
    mask = (dot > DOT_THRESHOLD)

    # Linear interpolation for close quaternions
    q_linear = q1 + t * (q2 - q1)
    q_linear = F.normalize(q_linear, p=2, dim=-1)

    # Spherical interpolation for distant quaternions
    theta = torch.acos(torch.clamp(dot, -1.0, 1.0))
    sin_theta = torch.sin(theta)

    # Avoid division by zero
    sin_theta = torch.clamp(sin_theta, min=1e-8)

    w1 = torch.sin((1.0 - t) * theta) / sin_theta
    w2 = torch.sin(t * theta) / sin_theta

    q_slerp = w1 * q1 + w2 * q2
    q_slerp = F.normalize(q_slerp, p=2, dim=-1)

    # Use linear for close, slerp for distant
    q_result = torch.where(mask, q_linear, q_slerp)

    return q_result

def integrate_gyro_pypose(gyro, dt, R0):
    """
    Integrate gyroscope to get rotation using PyPose.

    Args:
        gyro: [B, T, 3] or [T, 3] gyroscope (rad/s)
        dt: [B, T] or [T] time steps
        R0: [B, 4] or [4] initial rotation quaternion

    Returns:
        rot: [B, T+1, 4] or [T+1, 4] rotation quaternions
    """
    # Handle batch dimension
    if gyro.dim() == 2:  # [T, 3]
        gyro = gyro.unsqueeze(0)  # [1, T, 3]
        dt = dt.unsqueeze(0)  # [1, T]
        R0 = R0.unsqueeze(0)  # [1, 4]
        squeeze_output = True
    else:
        squeeze_output = False

    B, T = gyro.shape[:2]

    # Convert to SO3 increments
    theta = gyro * dt.unsqueeze(-1)  # [B, T, 3]
    delta = pp.so3(theta).Exp()  # [B, T] SO3

    # Cumulative product
    Rrel = pp.cumprod(delta, dim=1, left=False)  # [B, T] SO3

    # Apply initial rotation
    R0_so3 = pp.SO3(R0)  # [B] SO3
    R0_expanded = R0_so3.unsqueeze(1)  # [B, 1] SO3
    Rseq = torch.cat([R0_expanded.tensor(), (R0_expanded * Rrel).tensor()], dim=1)  # [B, T+1, 4]

    if squeeze_output:
        return Rseq.squeeze(0)  # [T+1, 4]
    return Rseq


def variance_weighted_fusion(gyro_right, gyro_left):
    """
    Optimal MMSE fusion using inverse variance weighting.

    Under independent Gaussian noise, this minimizes mean squared error.

    Reference: Kay, S. "Fundamentals of Statistical Signal Processing"

    Args:
        gyro_right, gyro_left: [B, T, 3] or [T, 3] gyroscope measurements

    Returns:
        gyro_fused: [B, T, 3] or [T, 3] optimally fused gyroscope
        weights: (w_right, w_left) normalized weights
    """
    # Compute variance (across time, per sequence)
    if gyro_right.dim() == 3:  # [B, T, 3]
        var_right = gyro_right.var(dim=1).mean(dim=-1, keepdim=True)  # [B, 1]
        var_left = gyro_left.var(dim=1).mean(dim=-1, keepdim=True)  # [B, 1]
    else:  # [T, 3]
        var_right = gyro_right.var(dim=0).mean()  # scalar
        var_left = gyro_left.var(dim=0).mean()  # scalar

    # Inverse variance weighting (MMSE optimal)
    w_right = 1.0 / (var_right + 1e-8)
    w_left = 1.0 / (var_left + 1e-8)
    w_total = w_right + w_left

    # Normalized weights
    w_right_norm = w_right / w_total
    w_left_norm = w_left / w_total

    # Weighted fusion
    if gyro_right.dim() == 3:
        gyro_fused = w_right_norm.unsqueeze(-1) * gyro_right + w_left_norm.unsqueeze(-1) * gyro_left
    else:
        gyro_fused = w_right_norm * gyro_right + w_left_norm * gyro_left

    return gyro_fused, (w_right_norm, w_left_norm)


def tilt_correction_kalman_gain(rot_gyro, accel, var_gyro, var_accel):
    """
    Tilt correction using Kalman-derived gain from sensor variances.

    The gain K = var_accel / (var_gyro + var_accel) is the steady-state Kalman gain
    for complementary filtering.

    Reference: Madgwick et al. "Estimation of IMU and MARG orientation using..."

    Args:
        rot_gyro: [B, T+1, 4] or [T+1, 4] rotation from gyro integration
        accel: [B, T, 3] or [T, 3] accelerometer measurements
        var_gyro: gyroscope variance
        var_accel: accelerometer variance

    Returns:
        rot_corrected: [B, T+1, 4] or [T+1, 4] tilt-corrected rotation
    """
    # Handle dimensions
    if rot_gyro.dim() == 2:  # [T+1, 4]
        rot_gyro_batch = rot_gyro.unsqueeze(0)  # [1, T+1, 4]
        accel_batch = accel.unsqueeze(0)  # [1, T, 3]
        squeeze_output = True
    else:
        rot_gyro_batch = rot_gyro
        accel_batch = accel
        squeeze_output = False

    # Pad accelerometer to match rotation length
    accel_padded = torch.cat([accel_batch[:, 0:1, :], accel_batch], dim=1)  # [B, T+1, 3]

    # Convert to SO3
    R = pp.SO3(rot_gyro_batch)  # [B, T+1] SO3

    # Expected gravity in world frame (CPF: Y-up, so gravity points in -Y direction)
    gravity_world = torch.tensor([0., -9.81, 0.], device=accel.device, dtype=accel.dtype)

    # Measured gravity direction (normalized accelerometer)
    accel_mag = accel_padded.norm(dim=-1, keepdim=True)  # [B, T+1, 1]
    gravity_measured = accel_padded / (accel_mag + 1e-8)  # [B, T+1, 3]

    # Expected gravity direction in body frame
    gravity_expected = R.Inv() @ gravity_world  # [B, T+1, 3]
    gravity_expected = gravity_expected / (gravity_expected.norm(dim=-1, keepdim=True) + 1e-8)

    # Compute tilt error (cross product gives rotation axis, dot product gives angle)
    axis = torch.cross(gravity_expected, gravity_measured, dim=-1)  # [B, T+1, 3]
    cos_angle = (gravity_expected * gravity_measured).sum(dim=-1, keepdim=True)  # [B, T+1, 1]
    cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
    angle = torch.acos(cos_angle)  # [B, T+1, 1]

    # Kalman gain (from sensor variances)
    # K = var_gyro / (var_gyro + var_accel)
    # Higher gyro variance → trust accelerometer more → higher K
    # Higher accel variance → trust gyro more → lower K
    K = var_gyro / (var_gyro + var_accel + 1e-8)

    # Reshape K for broadcasting
    # If K is a scalar, unsqueeze to [1, 1, 1]
    # If K is [B], unsqueeze to [B, 1, 1]
    if K.dim() == 0:  # Scalar
        K = K.view(1, 1, 1)  # [1, 1, 1] to broadcast with [B, T+1, 3]
    else:  # [B]
        K = K.unsqueeze(1).unsqueeze(1)  # [B, 1, 1] to broadcast with [B, T+1, 3]

    # Apply correction with Kalman gain
    axis_normalized = axis / (axis.norm(dim=-1, keepdim=True) + 1e-8)
    correction_vec = K * angle * axis_normalized  # [B, T+1, 3]

    # Convert to SO3 and apply correction
    correction_rot = pp.so3(correction_vec).Exp()  # [B, T+1] SO3
    R_corrected = R * correction_rot  # [B, T+1] SO3

    if squeeze_output:
        return R_corrected.tensor().squeeze(0)  # [T+1, 4]
    return R_corrected.tensor()  # [B, T+1, 4]


def preprocess_dual_imu(data, mode='right_only'):
    """
    Preprocess dual-IMU data based on specified mode.

    Args:
        data: Dict containing 'acc', 'gyro', 'rot', and optionally 'acc_left', 'gyro_left', 'rot_left'
        mode: One of the following:
            CPF-frame modes: 'right_only', 'average', 'rot_slerp', 'concat', 'avg_plus_diff'
            Body-frame modes: 'concat_body_no_left_rot', 'concat_body_with_left_rot',
                            'avg_plus_diff_cross_frame', 'avg_plus_diff_with_rot',
                            'weighted_concat_body', 'rotation_features_only',
                            'dual_encoder_no_left_rot', 'dual_encoder_with_left_rot',
                            'separate_temporal_cnn', 'relative_rotation_explicit',
                            'spatial_attention_fusion'

    Returns:
        processed_data: Dict with preprocessed 'acc', 'gyro', or dict with 'cpf' and 'body' keys for dual encoder
        processed_rot: Preprocessed rotation tensor
    """

    if mode == 'right_only':
        # Baseline: no preprocessing, use right IMU only
        return data, data['rot']

    # Check if left IMU data is available
    if 'acc_left' not in data or 'gyro_left' not in data:
        print(f"Warning: dual_imu_mode='{mode}' but left IMU data not found. Falling back to right_only.")
        return data, data['rot']

    if mode == 'average':
        # Average acceleration and gyroscope
        data_out = data.copy()
        data_out['acc'] = (data['acc'] + data['acc_left']) / 2.0
        data_out['gyro'] = (data['gyro'] + data['gyro_left']) / 2.0
        return data_out, data['rot']  # Keep original rotation

    elif mode == 'rot_slerp':
        # Average IMU + SLERP rotation
        data_out = data.copy()
        data_out['acc'] = (data['acc'] + data['acc_left']) / 2.0
        data_out['gyro'] = (data['gyro'] + data['gyro_left']) / 2.0

        if 'rot_left' in data:
            rot_avg = quaternion_slerp(data['rot'], data['rot_left'], t=0.5)
        else:
            print(f"Warning: rot_left not found, using rot only")
            rot_avg = data['rot']

        return data_out, rot_avg

    elif mode == 'concat':
        # Concatenate both IMUs: [acc_R, acc_L, gyro_R, gyro_L]
        data_out = data.copy()
        acc_concat = torch.cat([data['acc'], data['acc_left']], dim=-1)  # (B, T, 6)
        gyro_concat = torch.cat([data['gyro'], data['gyro_left']], dim=-1)  # (B, T, 6)
        data_out['acc'] = acc_concat
        data_out['gyro'] = gyro_concat

        # Concatenate rotations: [rot_R, rot_L]
        if 'rot_left' in data:
            rot_concat = torch.cat([data['rot'], data['rot_left']], dim=-1)  # (B, T, 8)
        else:
            print(f"Warning: rot_left not found, using rot only (duplicated)")
            rot_concat = torch.cat([data['rot'], data['rot']], dim=-1)

        return data_out, rot_concat

    elif mode == 'avg_plus_diff':
        # Consensus + disagreement features
        data_out = data.copy()

        # IMU: average + difference
        acc_avg = (data['acc'] + data['acc_left']) / 2.0
        acc_diff = data['acc'] - data['acc_left']
        gyro_avg = (data['gyro'] + data['gyro_left']) / 2.0
        gyro_diff = data['gyro'] - data['gyro_left']

        data_out['acc'] = torch.cat([acc_avg, acc_diff], dim=-1)  # (B, T, 6)
        data_out['gyro'] = torch.cat([gyro_avg, gyro_diff], dim=-1)  # (B, T, 6)

        # Rotation: average + relative
        if 'rot_left' in data:
            rot_avg = quaternion_slerp(data['rot'], data['rot_left'], t=0.5)  # (B, T, 4)
            rot_R_inv = quaternion_inverse(data['rot'])
            rot_rel = quaternion_multiply(data['rot_left'], rot_R_inv)  # rot_L * rot_R^(-1)
            rot_concat = torch.cat([rot_avg, rot_rel], dim=-1)  # (B, T, 8)
        else:
            print(f"Warning: rot_left not found, using identity for relative rotation")
            rot_avg = data['rot']
            rot_identity = torch.zeros_like(data['rot'])
            rot_identity[..., 3] = 1.0  # [0, 0, 0, 1]
            rot_concat = torch.cat([rot_avg, rot_identity], dim=-1)

        return data_out, rot_concat

    elif mode == 'variance_weighted':
        # MMSE-optimal fusion using inverse variance weighting
        # Reference: Kay, "Fundamentals of Statistical Signal Processing"
        gyro_fused, (w_r, w_l) = variance_weighted_fusion(data['gyro'], data['gyro_left'])

        # Also fuse accelerometers the same way
        accel_fused, _ = variance_weighted_fusion(data['acc'], data['acc_left'])

        data_out = data.copy()
        data_out['gyro'] = gyro_fused
        data_out['acc'] = accel_fused

        # Integrate fused gyro for rotation
        if hasattr(data['rot'], 'shape') and 'dt' in data:
            # dt has T+1 elements, gyro has T elements - slice dt to match
            T = gyro_fused.shape[1]
            dt_sliced = data['dt'][:, :T] if data['dt'].dim() == 2 else data['dt'][:T]
            rot_fused = integrate_gyro_pypose(gyro_fused, dt_sliced, data['rot'][:, 0, :])
            # Slice to match IMU length [B, T+1, 4] -> [B, T, 4]
            rot_fused = rot_fused[:, :-1, :]
        else:
            print(f"Warning: Cannot integrate gyro, using original rotation")
            rot_fused = data['rot']

        return data_out, rot_fused

    elif mode == 'variance_tilt':
        # Variance-weighted fusion + tilt correction with Kalman gain
        # Gain derived from sensor variances (steady-state Kalman filter)
        gyro_fused, (w_r, w_l) = variance_weighted_fusion(data['gyro'], data['gyro_left'])

        # Compute variances for Kalman gain
        var_gyro = data['gyro'].var(dim=1).mean(dim=-1, keepdim=True)  # [B, 1]
        var_accel = data['acc'].var(dim=1).mean(dim=-1, keepdim=True)  # [B, 1]

        # Integrate gyro
        if hasattr(data['rot'], 'shape') and 'dt' in data:
            # dt has T+1 elements, gyro has T elements - slice dt to match
            T = gyro_fused.shape[1]
            dt_sliced = data['dt'][:, :T] if data['dt'].dim() == 2 else data['dt'][:T]
            rot_gyro = integrate_gyro_pypose(gyro_fused, dt_sliced, data['rot'][:, 0, :])

            # Apply tilt correction with Kalman gain (returns [B, T+1, 4])
            rot_corrected = tilt_correction_kalman_gain(rot_gyro, data['acc'], var_gyro, var_accel)
            # Slice to match IMU length [B, T+1, 4] -> [B, T, 4]
            rot_corrected = rot_corrected[:, :-1, :]
        else:
            print(f"Warning: Cannot integrate gyro, using original rotation")
            rot_corrected = data['rot']

        data_out = data.copy()
        data_out['gyro'] = gyro_fused

        return data_out, rot_corrected

    elif mode == 'dual_accel_tilt':
        # Variance-weighted fusion for BOTH gyro and accel + tilt correction
        # This should be safe for both Nymeria and Aria
        gyro_fused, (w_gyro_r, w_gyro_l) = variance_weighted_fusion(data['gyro'], data['gyro_left'])
        accel_fused, (w_accel_r, w_accel_l) = variance_weighted_fusion(data['acc'], data['acc_left'])

        # Compute variances for Kalman gain
        var_gyro = data['gyro'].var(dim=1).mean(dim=-1, keepdim=True)  # [B, 1]
        # Use fused accelerometer variance
        if accel_fused.dim() == 3:
            var_accel = accel_fused.var(dim=1).mean(dim=-1, keepdim=True)  # [B, 1]
        else:
            var_accel = accel_fused.var(dim=0).mean()

        # Integrate gyro
        if hasattr(data['rot'], 'shape') and 'dt' in data:
            # dt has T+1 elements, gyro has T elements - slice dt to match
            T = gyro_fused.shape[1]
            dt_sliced = data['dt'][:, :T] if data['dt'].dim() == 2 else data['dt'][:T]
            rot_gyro = integrate_gyro_pypose(gyro_fused, dt_sliced, data['rot'][:, 0, :])

            # Apply tilt correction with fused accelerometer (returns [B, T+1, 4])
            rot_corrected = tilt_correction_kalman_gain(rot_gyro, accel_fused, var_gyro, var_accel)
            # Slice to match IMU length [B, T+1, 4] -> [B, T, 4]
            rot_corrected = rot_corrected[:, :-1, :]
        else:
            print(f"Warning: Cannot integrate gyro, using original rotation")
            rot_corrected = data['rot']

        data_out = data.copy()
        data_out['gyro'] = gyro_fused
        data_out['acc'] = accel_fused

        return data_out, rot_corrected

    # ===== Body-frame ablation modes (left IMU in body/device frame, right in CPF) =====
    elif mode == 'concat_body_no_left_rot':
        # Concatenate right (CPF) and left (body) IMU, no left rotation
        # Right IMU in CPF, Left IMU in body frame
        data_out = data.copy()
        acc_concat = torch.cat([data['acc'], data['acc_left']], dim=-1)  # (B, T, 6)
        gyro_concat = torch.cat([data['gyro'], data['gyro_left']], dim=-1)  # (B, T, 6)
        data_out['acc'] = acc_concat
        data_out['gyro'] = gyro_concat
        # Use only right rotation
        return data_out, data['rot']

    elif mode == 'concat_body_with_left_rot':
        # Concatenate right (CPF) and left (body) IMU with both rotations
        data_out = data.copy()
        acc_concat = torch.cat([data['acc'], data['acc_left']], dim=-1)  # (B, T, 6)
        gyro_concat = torch.cat([data['gyro'], data['gyro_left']], dim=-1)  # (B, T, 6)
        data_out['acc'] = acc_concat
        data_out['gyro'] = gyro_concat

        # Concatenate both rotations (extract quaternions from PyPose SO3 if needed)
        if 'rot_left' not in data:
            raise ValueError(f"Mode '{mode}' requires rot_left but it was not found in data. "
                           f"Ensure left IMU is loaded with gyro integration enabled.")

        import pypose as pp
        # Extract quaternions from PyPose SO3 tensors
        rot_right = data['rot'].quaternion() if isinstance(data['rot'], pp.LieTensor) else data['rot']
        rot_left = data['rot_left'].quaternion() if isinstance(data['rot_left'], pp.LieTensor) else data['rot_left']
        rot_concat = torch.cat([rot_right, rot_left], dim=-1)  # (B, T, 8)

        return data_out, rot_concat

    elif mode == 'avg_plus_diff_cross_frame':
        # Cross-frame features: raw values + spatial difference (even across frames)
        # Let model learn the transformation from the difference pattern
        data_out = data.copy()

        # Compute spatial differences (cross-frame)
        acc_diff = data['acc'] - data['acc_left']  # CPF - body frame
        gyro_diff = data['gyro'] - data['gyro_left']

        # Concatenate: [right, left, diff]
        data_out['acc'] = torch.cat([data['acc'], data['acc_left'], acc_diff], dim=-1)  # (B, T, 9)
        data_out['gyro'] = torch.cat([data['gyro'], data['gyro_left'], gyro_diff], dim=-1)  # (B, T, 9)

        # Use only right rotation
        return data_out, data['rot']

    elif mode == 'avg_plus_diff_with_rot':
        # Same as avg_plus_diff_cross_frame but include both rotations
        data_out = data.copy()

        acc_diff = data['acc'] - data['acc_left']
        gyro_diff = data['gyro'] - data['gyro_left']

        data_out['acc'] = torch.cat([data['acc'], data['acc_left'], acc_diff], dim=-1)  # (B, T, 9)
        data_out['gyro'] = torch.cat([data['gyro'], data['gyro_left'], gyro_diff], dim=-1)  # (B, T, 9)

        # Concatenate both rotations + relative rotation
        if 'rot_left' not in data:
            raise ValueError(f"Mode '{mode}' requires rot_left but it was not found in data. "
                           f"Ensure left IMU is loaded with gyro integration enabled.")

        import pypose as pp
        # Extract quaternions from PyPose SO3 tensors
        rot_right = data['rot'].quaternion() if isinstance(data['rot'], pp.LieTensor) else data['rot']
        rot_left = data['rot_left'].quaternion() if isinstance(data['rot_left'], pp.LieTensor) else data['rot_left']

        rot_R_inv = quaternion_inverse(rot_right)
        rot_rel = quaternion_multiply(rot_left, rot_R_inv)  # rot_L * rot_R^(-1)
        rot_concat = torch.cat([rot_right, rot_left, rot_rel], dim=-1)  # (B, T, 12)

        return data_out, rot_concat

    elif mode == 'weighted_concat_body':
        # Variance-weighted features before concatenation
        # Compute variance-based weights but keep signals separate
        if data['gyro'].dim() == 3:  # [B, T, 3]
            var_right = data['gyro'].var(dim=1).mean(dim=-1, keepdim=True).unsqueeze(-1)  # [B, 1, 1]
            var_left = data['gyro_left'].var(dim=1).mean(dim=-1, keepdim=True).unsqueeze(-1)
        else:
            var_right = data['gyro'].var(dim=0).mean()
            var_left = data['gyro_left'].var(dim=0).mean()

        # Normalize weights
        w_total = var_right + var_left + 1e-8
        w_right = var_right / w_total
        w_left = var_left / w_total

        data_out = data.copy()
        # Concatenate weighted signals
        acc_concat = torch.cat([w_right * data['acc'], w_left * data['acc_left']], dim=-1)
        gyro_concat = torch.cat([w_right * data['gyro'], w_left * data['gyro_left']], dim=-1)
        data_out['acc'] = acc_concat
        data_out['gyro'] = gyro_concat

        return data_out, data['rot']

    elif mode == 'rotation_features_only':
        # Emphasize rotation features: concat IMU with difference features
        # to match the 9D input dimension expected by the model
        data_out = data.copy()

        # Compute spatial differences
        acc_diff = data['acc'] - data['acc_left']
        gyro_diff = data['gyro'] - data['gyro_left']

        # Concatenate: [right, left, diff] to get 9D
        data_out['acc'] = torch.cat([data['acc'], data['acc_left'], acc_diff], dim=-1)  # (B, T, 9)
        data_out['gyro'] = torch.cat([data['gyro'], data['gyro_left'], gyro_diff], dim=-1)  # (B, T, 9)

        if 'rot_left' not in data:
            raise ValueError(f"Mode '{mode}' requires rot_left but it was not found in data. "
                           f"Ensure left IMU is loaded with gyro integration enabled.")

        import pypose as pp
        # Extract quaternions from PyPose SO3 tensors
        rot_right = data['rot'].quaternion() if isinstance(data['rot'], pp.LieTensor) else data['rot']
        rot_left = data['rot_left'].quaternion() if isinstance(data['rot_left'], pp.LieTensor) else data['rot_left']

        # Compute relative rotation (spatial relationship)
        rot_R_inv = quaternion_inverse(rot_right)
        rot_rel = quaternion_multiply(rot_left, rot_R_inv)

        # Concatenate: [rot_right, rot_left, rot_relative]
        rot_concat = torch.cat([rot_right, rot_left, rot_rel], dim=-1)  # (B, T, 12)

        return data_out, rot_concat

    elif mode == 'dual_encoder_no_left_rot':
        # Return dict format for dual encoder architecture
        # Separate data for CPF and body frame encoders
        data_out = {
            'cpf': {
                'acc': data['acc'],      # Right IMU in CPF (3D)
                'gyro': data['gyro'],    # Right IMU in CPF (3D)
            },
            'body': {
                'acc': data['acc_left'],  # Left IMU in body (3D)
                'gyro': data['gyro_left'], # Left IMU in body (3D)
            },
            # Preserve dt and other metadata
            'dt': data.get('dt'),
            'baro': data.get('baro'),
            'mag': data.get('mag'),
            'ts': data.get('ts'),
            'gt_rot': data.get('gt_rot'),
        }
        # Use only right rotation
        return data_out, data['rot']

    elif mode == 'dual_encoder_with_left_rot':
        # Dual encoder with both rotations
        import pypose as pp
        # Extract quaternions from PyPose SO3 tensors
        rot_right = data['rot'].quaternion() if isinstance(data['rot'], pp.LieTensor) else data['rot']

        data_out = {
            'cpf': {
                'acc': data['acc'],
                'gyro': data['gyro'],
                'rot': rot_right,      # Right rotation (4D)
            },
            'body': {
                'acc': data['acc_left'],
                'gyro': data['gyro_left'],
                'rot': None,  # Will be set below
            },
            # Preserve dt and other metadata
            'dt': data.get('dt'),
            'baro': data.get('baro'),
            'mag': data.get('mag'),
            'ts': data.get('ts'),
            'gt_rot': data.get('gt_rot'),
        }

        # Concatenate rotations for model
        if 'rot_left' not in data:
            raise ValueError(f"Mode '{mode}' requires rot_left but it was not found in data. "
                           f"Ensure left IMU is loaded with gyro integration enabled.")

        rot_left = data['rot_left'].quaternion() if isinstance(data['rot_left'], pp.LieTensor) else data['rot_left']
        data_out['body']['rot'] = rot_left
        rot_concat = torch.cat([rot_right, rot_left], dim=-1)

        return data_out, rot_concat

    elif mode == 'separate_temporal_cnn':
        # Similar to dual encoder but explicitly for separate temporal processing
        # Return as dict for model to process with separate temporal layers
        data_out = {
            'cpf': {
                'acc': data['acc'],
                'gyro': data['gyro'],
            },
            'body': {
                'acc': data['acc_left'],
                'gyro': data['gyro_left'],
            },
            # Preserve dt and other metadata
            'dt': data.get('dt'),
            'baro': data.get('baro'),
            'mag': data.get('mag'),
            'ts': data.get('ts'),
            'gt_rot': data.get('gt_rot'),
        }

        # Include both rotations
        if 'rot_left' not in data:
            raise ValueError(f"Mode '{mode}' requires rot_left but it was not found in data. "
                           f"Ensure left IMU is loaded with gyro integration enabled.")

        import pypose as pp
        rot_right = data['rot'].quaternion() if isinstance(data['rot'], pp.LieTensor) else data['rot']
        rot_left = data['rot_left'].quaternion() if isinstance(data['rot_left'], pp.LieTensor) else data['rot_left']
        rot_concat = torch.cat([rot_right, rot_left], dim=-1)

        return data_out, rot_concat

    elif mode == 'relative_rotation_explicit':
        # Explicitly encode relative rotation between frames as primary feature
        data_out = data.copy()
        acc_concat = torch.cat([data['acc'], data['acc_left']], dim=-1)
        gyro_concat = torch.cat([data['gyro'], data['gyro_left']], dim=-1)
        data_out['acc'] = acc_concat
        data_out['gyro'] = gyro_concat

        if 'rot_left' not in data:
            raise ValueError(f"Mode '{mode}' requires rot_left but it was not found in data. "
                           f"Ensure left IMU is loaded with gyro integration enabled.")

        import pypose as pp
        # Extract quaternions from PyPose SO3 tensors
        rot_right = data['rot'].quaternion() if isinstance(data['rot'], pp.LieTensor) else data['rot']
        rot_left = data['rot_left'].quaternion() if isinstance(data['rot_left'], pp.LieTensor) else data['rot_left']

        # Relative rotation: R_left_to_right = rot_R * rot_L^(-1)
        rot_L_inv = quaternion_inverse(rot_left)
        rot_rel = quaternion_multiply(rot_right, rot_L_inv)  # rot_R * rot_L^(-1)

        # Use relative rotation as primary, with right as reference
        rot_concat = torch.cat([rot_right, rot_rel], dim=-1)  # (B, T, 8)

        return data_out, rot_concat

    elif mode == 'spatial_attention_fusion':
        # Learnable fusion: concat all features and let model learn spatial attention
        data_out = data.copy()

        # Concatenate all available features
        acc_concat = torch.cat([data['acc'], data['acc_left']], dim=-1)
        gyro_concat = torch.cat([data['gyro'], data['gyro_left']], dim=-1)

        # Add difference features as attention cues
        acc_diff = data['acc'] - data['acc_left']
        gyro_diff = data['gyro'] - data['gyro_left']

        data_out['acc'] = torch.cat([acc_concat, acc_diff], dim=-1)  # (B, T, 9)
        data_out['gyro'] = torch.cat([gyro_concat, gyro_diff], dim=-1)  # (B, T, 9)

        # Full rotation info for spatial attention
        if 'rot_left' not in data:
            raise ValueError(f"Mode '{mode}' requires rot_left but it was not found in data. "
                           f"Ensure left IMU is loaded with gyro integration enabled.")

        import pypose as pp
        # Extract quaternions from PyPose SO3 tensors
        rot_right = data['rot'].quaternion() if isinstance(data['rot'], pp.LieTensor) else data['rot']
        rot_left = data['rot_left'].quaternion() if isinstance(data['rot_left'], pp.LieTensor) else data['rot_left']

        rot_R_inv = quaternion_inverse(rot_right)
        rot_rel = quaternion_multiply(rot_left, rot_R_inv)
        rot_concat = torch.cat([rot_right, rot_left, rot_rel], dim=-1)  # (B, T, 12)

        return data_out, rot_concat

    else:
        raise ValueError(f"Unknown dual_imu_mode: '{mode}'. "
                        f"Expected one of ['right_only', 'average', 'rot_slerp', 'concat', 'avg_plus_diff', "
                        f"'variance_weighted', 'variance_tilt', 'dual_accel_tilt', "
                        f"'concat_body_no_left_rot', 'concat_body_with_left_rot', "
                        f"'avg_plus_diff_cross_frame', 'avg_plus_diff_with_rot', "
                        f"'weighted_concat_body', 'rotation_features_only', "
                        f"'dual_encoder_no_left_rot', 'dual_encoder_with_left_rot', "
                        f"'separate_temporal_cnn', 'relative_rotation_explicit', 'spatial_attention_fusion']")


# ===== For use in training loop =====
def preprocess_training_batch(data, label, confs):
    """
    Preprocess a training batch based on config.

    Args:
        data: Input data dict
        label: Label dict
        confs: Training config with dual_imu_mode

    Returns:
        data: Preprocessed data
        rot: Preprocessed rotation for model input
    """
    dual_imu_mode = getattr(confs, 'dual_imu_mode', 'right_only')
    use_raw_quat = getattr(confs, 'use_raw_quat', False)

    # Preprocess dual-IMU data
    data, rot = preprocess_dual_imu(data, mode=dual_imu_mode)

    # Handle rotation format (from label if not preprocessed)
    if dual_imu_mode in ['right_only', 'average']:
        # Use original rotation extraction logic
        if use_raw_quat:
            if hasattr(label['gt_rot'], 'quaternion'):
                rot = label['gt_rot'][:,:-1,:].quaternion()
            else:
                rot = label['gt_rot'][:,:-1,:]
        else:
            if hasattr(label['gt_rot'], 'Log'):
                import pypose as pp
                rot = label['gt_rot'][:,:-1,:].Log().tensor()
            else:
                import pypose as pp
                rot = pp.SO3(label['gt_rot'][:,:-1,:]).Log().tensor()

    return data, rot
