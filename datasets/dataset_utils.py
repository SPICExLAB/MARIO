import torch
import math
import pypose as pp
import numpy as np
import gc


def imu_seq_collate(data):
    acc = torch.stack([d["acc"] for d in data])
    gyro = torch.stack([d["gyro"] for d in data])

    gt_pos = torch.stack([d["gt_pos"] for d in data])
    
    # Handle gt_rot specially to preserve PyPose tensors
    if len(data) > 0 and isinstance(data[0]["gt_rot"], pp.LieTensor):
        # Use torch.stack and then convert back to PyPose
        gt_rot_tensors = torch.stack([d["gt_rot"].tensor() for d in data])
        gt_rot = pp.SO3(gt_rot_tensors)
    else:
        gt_rot = torch.stack([d["gt_rot"] for d in data])
        
    gt_vel = torch.stack([d["gt_vel"] for d in data])

    init_pos = torch.stack([d["init_pos"] for d in data])
    
    # For init_rot, always use torch.stack since PyPose tensors will be converted
    init_rot = torch.stack([d["init_rot"] for d in data])
        
    init_vel = torch.stack([d["init_vel"] for d in data])

    dt = torch.stack([d["dt"] for d in data])

    return {
        "dt": dt,
        "acc": acc,
        "gyro": gyro,
        "gt_pos": gt_pos,
        "gt_vel": gt_vel,
        "gt_rot": gt_rot,
        "init_pos": init_pos,
        "init_vel": init_vel,
        "init_rot": init_rot,
    }


def custom_collate(data):
    dt = torch.stack([d["dt"] for d in data])
    acc = torch.stack([d["acc"] for d in data])
    gyro = torch.stack([d["gyro"] for d in data])
    rot = torch.stack([d["rot"] for d in data])

    gt_pos = torch.stack([d["gt_pos"] for d in data])
    
    # Handle gt_rot specially to preserve PyPose tensors
    if len(data) > 0 and isinstance(data[0]["gt_rot"], pp.LieTensor):
        # Use torch.stack and then convert back to PyPose
        gt_rot_tensors = torch.stack([d["gt_rot"].tensor() for d in data])
        gt_rot = pp.SO3(gt_rot_tensors)
    else:
        gt_rot = torch.stack([d["gt_rot"] for d in data])
        
    gt_vel = torch.stack([d["gt_vel"] for d in data])

    init_pos = torch.stack([d["init_pos"] for d in data])
    
    # For init_rot, always use torch.stack since PyPose tensors will be converted
    init_rot = torch.stack([d["init_rot"] for d in data])
        
    init_vel = torch.stack([d["init_vel"] for d in data])

    return (
        {
            "dt": dt,
            "acc": acc,
            "gyro": gyro,
            "rot": rot,
        },
        {
            "pos": init_pos,
            "vel": init_vel,
            "rot": init_rot,
        },
        {
            "gt_pos": gt_pos,
            "gt_vel": gt_vel,
            "gt_rot": gt_rot,
        },
    )

def motion_collate_data(data):
    # Filter out samples with mismatched sizes
    # Get the most common size for each field
    acc_sizes = [d['acc'].shape[0] for d in data]
    from collections import Counter
    target_size = Counter(acc_sizes).most_common(1)[0][0]

    # Filter data to only keep samples matching target size
    filtered_data = [d for d in data if d['acc'].shape[0] == target_size]

    if len(filtered_data) < len(data):
        print(f"Warning: Filtered out {len(data) - len(filtered_data)} samples with mismatched sizes")

    # Use filtered data for stacking
    timestamp = None
    timestamp = [d['timestamp'] for d in filtered_data if 'timestamp' in d]
    if timestamp:
        timestamp = torch.stack(timestamp)
    acc = torch.stack([d['acc'] for d in filtered_data])
    gyro = torch.stack([d['gyro'] for d in filtered_data])
    rot = torch.stack([d['rot'] for d in filtered_data])

    gt_pos = torch.stack([d["gt_pos"] for d in filtered_data])

    # For gt_rot, always use torch.stack since PyPose tensors will be converted
    gt_rot = torch.stack([d["gt_rot"] for d in filtered_data])

    gt_vel = torch.stack([d["gt_vel"] for d in filtered_data])

    init_pos = torch.stack([d["init_pos"] for d in filtered_data])

    # For init_rot, always use torch.stack since PyPose tensors will be converted
    init_rot = torch.stack([d["init_rot"] for d in filtered_data])

    init_vel = torch.stack([d["init_vel"] for d in filtered_data])

    dt = torch.stack([d['dt'] for d in filtered_data])

    baro = torch.stack([d['baro'] for d in filtered_data])

    mag = torch.stack([d['mag'] for d in filtered_data])

    # Prepare output dict
    output_data = {
        'ts': timestamp,
        "dt": dt,
        "acc": acc,
        "gyro": gyro,
        "rot": rot,
        "baro": baro,
        "mag": mag,
    }

    # Add left IMU data if available (optional)
    if filtered_data and 'acc_left' in filtered_data[0]:
        acc_left = torch.stack([d['acc_left'] for d in filtered_data])
        gyro_left = torch.stack([d['gyro_left'] for d in filtered_data])
        output_data['acc_left'] = acc_left
        output_data['gyro_left'] = gyro_left
    else:
        # Default empty tensors if not available
        acc_left = torch.zeros_like(acc)
        gyro_left = torch.zeros_like(gyro)
        output_data['acc_left'] = acc_left
        output_data['gyro_left'] = gyro_left

    if filtered_data and 'rot_left' in filtered_data[0]:
        rot_left = torch.stack([d['rot_left'] for d in filtered_data])
        output_data['rot_left'] = rot_left

    # Body joint labels for PoseNet training (optional)
    if filtered_data and 'joint_pos' in filtered_data[0]:
        output_data['joint_pos'] = torch.stack([d['joint_pos'] for d in filtered_data])
        output_data['joint_rot'] = torch.stack([d['joint_rot'] for d in filtered_data])

    return (
        output_data,
        {
            "pos": init_pos,
            "vel": init_vel,
            "rot": init_rot,
        },
        {
            "gt_pos": gt_pos,
            "gt_vel": gt_vel,
            "gt_rot": gt_rot,
        },
    )
    
def motion_collate(data, **kwargs):
    input_data, init_state, label = motion_collate_data(data)
    if len(kwargs) > 0:
        # TODO: Implement data augmentation if needed
        pass  
    return input_data, init_state, label

    
collate_fcs = {
    "base": custom_collate,
    'motion': motion_collate,
}

def truncate_and_free(d, keys, maximum_length):
    for key in keys:
        if key not in d:
            continue
        arr = d[key]  # hold old reference

        # Make a true copy with its own (smaller) storage
        if isinstance(arr, torch.Tensor):
            # If you don't need grads, ensure requires_grad=False
            if arr.requires_grad:
                arr = arr.detach()
            d[key] = arr[:maximum_length].clone()  # NEW storage
        elif isinstance(arr, np.ndarray):
            d[key] = arr[:maximum_length].copy()   # NEW storage
        else:
            # For lists or other sequences, slice creates a new list anyway
            d[key] = arr[:maximum_length]

        # Drop the only remaining reference to the big buffer
        del arr

    # Encourage Python to reclaim CPU RAM
    gc.collect()

    # If any were CUDA tensors, this actually returns free VRAM to CUDA allocator
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
def smoothing(x, kernel_size=101):
    kernel = np.ones(kernel_size) / kernel_size
    pad = kernel_size // 2
    x_padded = np.pad(x, (pad, pad), mode='reflect')
    smoothed = np.convolve(x_padded, kernel, mode='valid')
    return smoothed
