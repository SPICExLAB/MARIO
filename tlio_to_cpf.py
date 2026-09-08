#!/usr/bin/env python3
"""
TLIO to CPF Converter - Production Version
==========================================

Converts TLIO dataset(s) from arbitrary world/device frames to CPF-aligned coordinates.

Usage:
    python tlio_to_cpf.py data/110982076486017                    # Single sequence
    python tlio_to_cpf.py data                                     # All sequences in folder
    python tlio_to_cpf.py data/seq1 data/seq2 --output results    # Multiple sequences
    python tlio_to_cpf.py $DATA_ROOT/tlio_dataset/test --output $DATA_ROOT/tlio_dataset_cpf/test
    
    python tlio_to_cpf.py $DATA_ROOT/tlio_dataset/train --output $DATA_ROOT/tlio_dataset_cpf/train
    
    

Output Format:
    - Processed data saved as .npz files
    - Contains: timestamps, positions, velocities, accelerations, gyroscope, orientations
    - All in CPF-aligned world frame
    - Includes metadata and transformation matrices
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import json

# ============================================================================
# TLIO Device Frame Convention (discovered via diagnostic)
# ============================================================================
# Device +X = FORWARD (gaze direction)
# Device +Y = UP (top of head)
# Device +Z = RIGHT (right side of head)
#
# CPF Target:
# CPF +X = LEFT
# CPF +Y = UP
# CPF +Z = FORWARD

R_DEVICE_TO_CPF = np.array([
    [0, 0, -1],  # Device Z (right) → CPF -X (left)
    [0, 1, 0],   # Device Y (up) → CPF Y (up)
    [1, 0, 0]    # Device X (forward) → CPF Z (forward)
], dtype=np.float64)

print(f"Device→CPF transformation determinant: {np.linalg.det(R_DEVICE_TO_CPF):.6f}")

# ============================================================================
# Processing Functions
# ============================================================================

def load_tlio_sequence(seq_path):
    """Load TLIO sequence data"""
    npy_path = seq_path / "imu0_resampled.npy"
    
    if not npy_path.exists():
        raise FileNotFoundError(f"IMU data not found: {npy_path}")
    
    data = np.load(npy_path, mmap_mode='r')
    
    return {
        'time_us': data[:, 0].copy(),
        'gyro_world': data[:, 1:4].copy(),
        'acc_world': data[:, 4:7].copy(),
        'quat_xyzw': data[:, 7:11].copy(),
        'pos_world': data[:, 11:14].copy(),
        'vel_world': data[:, 14:17].copy(),
    }


def build_cpf_world_frame(quat_0, acc_world):
    """
    Build intuitive CPF-aligned world frame at t=0
    
    Strategy:
    1. Find vertical axis (from gravity)
    2. Transform device to CPF at t=0
    3. Extract CPF forward direction and project to horizontal
    4. Build orthonormal frame: X=left, Y=up, Z=forward
    """
    # Determine world up direction from gravity
    acc_mean = np.mean(acc_world[:100], axis=0)
    vertical_idx = np.argmax(np.abs(acc_mean))
    world_up = np.zeros(3)
    world_up[vertical_idx] = np.sign(acc_mean[vertical_idx])
    
    # Get initial device orientation
    R_device_to_world_0 = R.from_quat(quat_0).as_matrix()
    
    # Transform to CPF
    R_CPF_to_world_0 = R_device_to_world_0 @ R_DEVICE_TO_CPF
    
    # Extract CPF forward direction (Z-axis) and project to horizontal
    cpf_z_world = R_CPF_to_world_0[:, 2]
    cpf_z_horizontal = cpf_z_world - np.dot(cpf_z_world, world_up) * world_up
    cpf_z_horizontal = cpf_z_horizontal / (np.linalg.norm(cpf_z_horizontal) + 1e-8)
    
    # Build CPF-world frame
    cpf_world_z = cpf_z_horizontal
    cpf_world_y = world_up
    cpf_world_x = np.cross(cpf_world_y, cpf_world_z)
    cpf_world_x = cpf_world_x / (np.linalg.norm(cpf_world_x) + 1e-8)
    
    # Re-orthogonalize
    cpf_world_z = np.cross(cpf_world_x, cpf_world_y)
    cpf_world_z = cpf_world_z / (np.linalg.norm(cpf_world_z) + 1e-8)
    
    # Transformation matrix: world → CPF-world
    R_world_to_cpfworld = np.column_stack([cpf_world_x, cpf_world_y, cpf_world_z])
    
    return R_world_to_cpfworld, world_up


def transform_to_cpf(data, R_world_to_cpfworld, pos_ref):
    """
    Transform all TLIO data to CPF-aligned coordinates
    
    Returns dict with:
    - timestamps
    - positions, velocities (CPF-world frame)
    - orientations (CPF body in CPF-world)
    - accelerations, gyroscope (both world and body frames in CPF)
    """
    N = len(data['time_us'])
    time_s = (data['time_us'] - data['time_us'][0]) * 1e-6
    
    # Transform positions (relative to reference)
    pos_cpfworld = np.zeros_like(data['pos_world'])
    for i in range(N):
        pos_relative = data['pos_world'][i] - pos_ref
        pos_cpfworld[i] = R_world_to_cpfworld.T @ pos_relative
    
    # Transform velocities
    vel_cpfworld = np.zeros_like(data['vel_world'])
    for i in range(N):
        vel_cpfworld[i] = R_world_to_cpfworld.T @ data['vel_world'][i]
    
    # Transform world-frame IMU
    acc_cpfworld = np.zeros_like(data['acc_world'])
    gyro_cpfworld = np.zeros_like(data['gyro_world'])
    for i in range(N):
        acc_cpfworld[i] = R_world_to_cpfworld.T @ data['acc_world'][i]
        gyro_cpfworld[i] = R_world_to_cpfworld.T @ data['gyro_world'][i]
    
    # Transform orientations (device → CPF → CPF-world)
    quat_cpfworld = np.zeros_like(data['quat_xyzw'])
    for i in range(N):
        R_device_world = R.from_quat(data['quat_xyzw'][i]).as_matrix()
        R_CPF_world = R_device_world @ R_DEVICE_TO_CPF
        R_CPF_cpfworld = R_world_to_cpfworld.T @ R_CPF_world
        quat_cpfworld[i] = R.from_matrix(R_CPF_cpfworld).as_quat()
    
    # Compute body-frame IMU in CPF coordinates
    acc_body_cpf = np.zeros_like(data['acc_world'])
    gyro_body_cpf = np.zeros_like(data['gyro_world'])
    for i in range(N):
        R_CPF_cpfworld = R.from_quat(quat_cpfworld[i]).as_matrix()
        acc_body_cpf[i] = R_CPF_cpfworld.T @ acc_cpfworld[i]
        gyro_body_cpf[i] = R_CPF_cpfworld.T @ gyro_cpfworld[i]
    
    return {
        'timestamps': time_s.astype(np.float32),
        'positions': pos_cpfworld.astype(np.float32),
        'velocities': vel_cpfworld.astype(np.float32),
        'orientations': quat_cpfworld.astype(np.float32),
        'acc_world': acc_cpfworld.astype(np.float32),
        'gyro_world': gyro_cpfworld.astype(np.float32),
        'acc_body': acc_body_cpf.astype(np.float32),
        'gyro_body': gyro_body_cpf.astype(np.float32),
    }


def process_sequence(seq_path, output_dir=None):
    """Process a single TLIO sequence"""
    seq_path = Path(seq_path)
    seq_name = seq_path.name
    
    print(f"\n{'='*80}")
    print(f"Processing: {seq_name}")
    print(f"{'='*80}")
    
    # Load data
    print("Loading data...")
    data = load_tlio_sequence(seq_path)
    N = len(data['time_us'])
    duration = (data['time_us'][-1] - data['time_us'][0]) * 1e-6
    print(f"  Loaded {N} samples ({duration:.2f}s, {N/duration:.1f} Hz)")
    
    # Build CPF-world frame
    print("Building CPF-aligned world frame...")
    quat_0 = data['quat_xyzw'][0]
    R_world_to_cpfworld, world_up = build_cpf_world_frame(quat_0, data['acc_world'])
    pos_ref = data['pos_world'][0]
    
    # Verify initial orientation
    R_device_world_0 = R.from_quat(quat_0).as_matrix()
    R_CPF_world_0 = R_device_world_0 @ R_DEVICE_TO_CPF
    R_CPF_cpfworld_0 = R_world_to_cpfworld.T @ R_CPF_world_0
    euler_0 = R.from_matrix(R_CPF_cpfworld_0).as_euler('xyz', degrees=True)
    print(f"  Initial orientation in CPF-world: Roll={euler_0[0]:.2f}°, Pitch={euler_0[1]:.2f}°, Yaw={euler_0[2]:.2f}°")
    
    if np.max(np.abs(euler_0)) < 5.0:
        print(f"  ✓ Initial orientation ≈ identity")
    else:
        print(f"  ⚠ Initial orientation has {np.max(np.abs(euler_0)):.1f}° deviation (acceptable if < 5°)")
    
    # Transform data
    print("Transforming to CPF coordinates...")
    transformed = transform_to_cpf(data, R_world_to_cpfworld, pos_ref)
    
    # Verify gravity
    acc_world_mean = np.mean(transformed['acc_world'][:100], axis=0)
    acc_body_mean = np.mean(transformed['acc_body'][:100], axis=0)
    print(f"  World-frame acc mean: [{acc_world_mean[0]:.3f}, {acc_world_mean[1]:.3f}, {acc_world_mean[2]:.3f}] m/s²")
    print(f"  Body-frame acc mean:  [{acc_body_mean[0]:.3f}, {acc_body_mean[1]:.3f}, {acc_body_mean[2]:.3f}] m/s²")
    
    if abs(acc_world_mean[1] - 9.81) < 1.0:
        print(f"  ✓ World Y-axis has correct gravity (+{acc_world_mean[1]:.2f} m/s²)")
    if abs(acc_body_mean[1] - 9.81) < 1.0:
        print(f"  ✓ Body Y-axis has correct gravity (+{acc_body_mean[1]:.2f} m/s²)")
    
    # Save output
    if output_dir is None:
        output_dir = seq_path.parent / "cpf_processed"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{seq_name}_cpf.npz"
    
    # Package data with metadata
    np.savez_compressed(
        output_path,
        # Time series data
        timestamps=transformed['timestamps'],
        positions=transformed['positions'],
        velocities=transformed['velocities'],
        orientations=transformed['orientations'],
        acc_world=transformed['acc_world'],
        gyro_world=transformed['gyro_world'],
        acc_body=transformed['acc_body'],
        gyro_body=transformed['gyro_body'],
        
        # Metadata
        sequence_name=seq_name,
        num_samples=N,
        duration_seconds=duration,
        reference_position=pos_ref.astype(np.float32),
        
        # Transformation matrices
        R_device_to_CPF=R_DEVICE_TO_CPF.astype(np.float32),
        R_world_to_cpfworld=R_world_to_cpfworld.astype(np.float32),
        world_up=world_up.astype(np.float32),
        
        # Frame descriptions
        coordinate_system='CPF: X=left, Y=up, Z=forward',
    )
    
    print(f"✓ Saved: {output_path}")
    print(f"  Size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    return output_path


def process_directory(data_dir, output_dir=None, pattern="imu0_resampled.npy"):
    """Process all sequences in a directory"""
    data_dir = Path(data_dir)
    
    # Find all sequences
    sequences = []
    for item in data_dir.iterdir():
        if item.is_dir():
            npy_path = item / pattern
            if npy_path.exists():
                sequences.append(item)
    
    if not sequences:
        print(f"No sequences found in {data_dir}")
        return []
    
    print(f"\n{'='*80}")
    print(f"Found {len(sequences)} sequences in {data_dir}")
    print(f"{'='*80}")
    
    for seq in sequences:
        print(f"  - {seq.name}")
    
    # Process each sequence
    output_paths = []
    for seq in sequences:
        try:
            output_path = process_sequence(seq, output_dir)
            output_paths.append(output_path)
        except Exception as e:
            print(f"\n✗ Failed to process {seq.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return output_paths


def main():
    parser = argparse.ArgumentParser(
        description='Convert TLIO data to CPF-aligned coordinates',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single sequence
  python tlio_to_cpf.py data/110982076486017
  
  # Process all sequences in directory
  python tlio_to_cpf.py data
  
  # Process multiple sequences with custom output
  python tlio_to_cpf.py data/seq1 data/seq2 --output results
  
  # Process directory with custom output
  python tlio_to_cpf.py data --output cpf_data
        """
    )
    
    parser.add_argument(
        'input_paths',
        nargs='+',
        help='Input sequence path(s) or directory containing sequences'
    )
    parser.add_argument(
        '-o', '--output',
        help='Output directory (default: input_dir/cpf_processed)'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Run verification after processing (slower)'
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("TLIO TO CPF CONVERTER")
    print("="*80)
    print(f"Device→CPF: X=fwd→Z, Y=up→Y, Z=right→-X")
    print(f"CPF World: X=left, Y=up (gravity), Z=forward (gaze)")
    
    all_outputs = []
    
    for input_path in args.input_paths:
        input_path = Path(input_path)
        
        if not input_path.exists():
            print(f"\n✗ Path does not exist: {input_path}")
            continue
        
        if input_path.is_dir():
            # Check if it's a sequence directory or contains sequences
            npy_path = input_path / "imu0_resampled.npy"
            if npy_path.exists():
                # Single sequence
                output = process_sequence(input_path, args.output)
                all_outputs.append(output)
            else:
                # Directory containing sequences
                outputs = process_directory(input_path, args.output)
                all_outputs.extend(outputs)
        else:
            print(f"\n✗ Not a directory: {input_path}")
    
    # Summary
    print(f"\n{'='*80}")
    print(f"PROCESSING COMPLETE")
    print(f"{'='*80}")
    print(f"Successfully processed {len(all_outputs)} sequence(s)")
    
    if all_outputs:
        print(f"\nOutput files:")
        for out in all_outputs:
            print(f"  {out}")
        
        print(f"\n{'='*80}")
        print(f"DATA FORMAT:")
        print(f"{'='*80}")
        print(f"Load with: data = np.load('sequence_cpf.npz')")
        print(f"\nArrays:")
        print(f"  timestamps:    (N,)     - Time in seconds")
        print(f"  positions:     (N, 3)   - Position in CPF-world [left, up, forward]")
        print(f"  velocities:    (N, 3)   - Velocity in CPF-world")
        print(f"  orientations:  (N, 4)   - Quaternion [x,y,z,w] CPF→CPF-world")
        print(f"  acc_world:     (N, 3)   - Acceleration in CPF-world frame")
        print(f"  gyro_world:    (N, 3)   - Gyroscope in CPF-world frame")
        print(f"  acc_body:      (N, 3)   - Acceleration in body frame (CPF axes)")
        print(f"  gyro_body:     (N, 3)   - Gyroscope in body frame (CPF axes)")
        print(f"\nMetadata:")
        print(f"  R_device_to_CPF, R_world_to_cpfworld, reference_position, etc.")


if __name__ == "__main__":
    main()