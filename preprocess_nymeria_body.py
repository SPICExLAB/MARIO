#!/usr/bin/env python3
"""
Corrected CPF-Aligned World Frame Processing Pipeline for Nymeria Dataset

Key corrections based on validation:
1. SLAM quaternion: NO transpose (represents T_device_world directly)
2. Device-to-CPF for frames: R_raw (no transpose)
3. Device-to-CPF for vectors: R_raw.T (with transpose)
4. Proper velocity transformation and integration
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial.transform import Rotation, Slerp
from tqdm import tqdm
import json
import pickle
import argparse
from multiprocessing import Pool, cpu_count
import torch

try:
    from projectaria_tools.core import data_provider
    PROJECTARIA_AVAILABLE = True
except ImportError:
    PROJECTARIA_AVAILABLE = False
    print("⚠️ ProjectAria tools not available")


class CPFWorldTransformer:
    """Transform Nymeria data to CPF-aligned world frame with corrected transformations"""
    
    def __init__(self, sequence_path: Path, skip_seconds: float = 30.0):
        self.sequence_path = sequence_path
        self.skip_seconds = skip_seconds
        
        # Paths (keeping the working path structure)
        self.vrs_path = sequence_path / "recording_head" / "data" / "motion.vrs"
        self.traj_path = sequence_path / "recording_head" / "mps" / "slam" / "closed_loop_trajectory.csv"
        self.calib_path = sequence_path / "recording_head" / "mps" / "slam" / "online_calibration.jsonl"
        
        # Transformations (to be computed)
        self.R_device_to_CPF_frames = None   # For coordinate frames
        self.R_device_to_CPF_vectors = None  # For vectors (IMU, velocity)
        self.R_world_to_cpfworld = None
        self.reference_time = None
        self.reference_position = None
        
    def load_calibrations(self):
        """Load device and IMU calibrations with correct understanding"""
        if not PROJECTARIA_AVAILABLE:
            raise RuntimeError("ProjectAria tools required")
            
        # Load device calibration
        self.provider = data_provider.create_vrs_data_provider(str(self.vrs_path))
        device_calibration = self.provider.get_device_calibration()
        T_device_CPF = device_calibration.get_transform_device_cpf()
        # Get raw matrix
        matrix_4x4 = T_device_CPF.to_matrix()
        R_raw = matrix_4x4[:3, :3]
        
        # CORRECTED: Different uses need different forms
        self.R_device_to_CPF_frames = R_raw      # For frame transformations (no transpose)
        self.R_device_to_CPF_vectors = R_raw.T   # For vector transformations (with transpose)
        breakpoint()
        
        # Load IMU calibration
        self.imu_calib = self._load_imu_calibration()
        
        print(f"✅ Loaded calibrations")
        print(f"   R_device_to_CPF_frames det: {np.linalg.det(self.R_device_to_CPF_frames):.6f}")
        print(f"   Using R_raw for frames, R_raw.T for vectors")
        
    def _load_imu_calibration(self):
        """Load IMU calibration from file"""
        if not self.calib_path.exists():
            print("⚠️ No IMU calibration found, using identity")
            return None
            
        calibrations = []
        with open(self.calib_path, 'r') as f:
            for line in f:
                calibrations.append(json.loads(line))
        
        for calib in calibrations:
            for imu_calib in calib.get('ImuCalibrations', []):
                if imu_calib['Label'] == 'imu-right':
                    T_Device_Imu = imu_calib['T_Device_Imu']
                    quat = T_Device_Imu['UnitQuaternion']
                    qw = quat[0]
                    qx, qy, qz = quat[1][0], quat[1][1], quat[1][2]
                    R_Device_Imu = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
                    
                    return {
                        'R_Device_Imu': R_Device_Imu,
                        'gyro_bias': np.array(imu_calib['Gyroscope']['Bias']['Offset']),
                        'accel_bias': np.array(imu_calib['Accelerometer']['Bias']['Offset']),
                    }
        return None
    
    def establish_cpf_world_frame(self, df: pd.DataFrame):
        """
        Establish CPF-aligned world frame at reference time
        CORRECTED: Use proper transformation chain
        """
        # Find reference time
        traj_times = df['tracking_timestamp_us'].values / 1e6
        
        # Get first IMU timestamp for reference
        imu_id = self.provider.get_stream_id_from_label("imu-right")
        first_sample = self.provider.get_imu_data_by_index(imu_id, 0)
        imu_start_time = first_sample.capture_timestamp_ns * 1e-9
        
        # Find trajectory index at reference time
        target_ref_time = imu_start_time + self.skip_seconds
        ref_idx = np.argmin(np.abs(traj_times - target_ref_time))
        self.reference_time = traj_times[ref_idx]
        
        # Get device orientation at reference
        quat_ref = df.iloc[ref_idx][['qw_world_device', 'qx_world_device', 
                                      'qy_world_device', 'qz_world_device']].values
        
        # CORRECTED: NO transpose for SLAM quaternion
        R_device_to_world_ref = Rotation.from_quat([quat_ref[1], quat_ref[2], 
                                                    quat_ref[3], quat_ref[0]]).as_matrix()
        
        # CORRECTED: Transform to CPF using frame transformation
        R_CPF_to_world_ref = R_device_to_world_ref @ self.R_device_to_CPF_frames
        
        # Verify CPF Y points up
        cpf_y_in_world = R_CPF_to_world_ref[:, 1]
        if cpf_y_in_world[2] < 0.7:
            print(f"⚠️ Warning: CPF Y not pointing up (Z={cpf_y_in_world[2]:.3f})")
        
        # Extract CPF forward direction and project to horizontal
        cpf_z_world = R_CPF_to_world_ref[:, 2]  # CPF Z-axis in world
        forward_horizontal = np.array([cpf_z_world[0], cpf_z_world[1], 0])
        forward_horizontal = forward_horizontal / (np.linalg.norm(forward_horizontal) + 1e-8)
        
        # Build CPF-aligned world frame
        cpfworld_z = forward_horizontal      # Forward (initial gaze)
        cpfworld_y = np.array([0, 0, 1])    # Up (gravity)
        cpfworld_x = np.cross(cpfworld_y, cpfworld_z) # Left
        cpfworld_x = cpfworld_x / np.linalg.norm(cpfworld_x)
        
        # This transforms FROM world TO CPF-world
        self.R_world_to_cpfworld = np.column_stack([cpfworld_x, cpfworld_y, cpfworld_z])
        
        # Store reference position
        ref_position = df.iloc[ref_idx][['tx_world_device', 
                                         'ty_world_device', 
                                         'tz_world_device']].values
        self.reference_position = np.array(ref_position, dtype=np.float64)
        
        print(f"✅ CPF-aligned world frame established at t={self.reference_time:.1f}s")
        print(f"   CPF Y in world Z: {cpf_y_in_world[2]:.3f} (should be ~1)")
        print(f"   Initial yaw = 0° by construction")
        
    def process_full_sequence(self, imu_hz=200, pose_hz=50):
        """Process entire sequence with corrected transformations
        
        Args:
            imu_hz: Sampling rate for IMU data (accel, gyro) - default 200Hz
            pose_hz: Sampling rate for pose data (orientation, position, velocity) - default 50Hz
        """
        
        print(f"\\n{'='*70}")
        print(f"🚀 PROCESSING: {self.sequence_path.name}")
        print(f"{'='*70}")
        
        # Check files exist
        for name, path in [("VRS", self.vrs_path), ("Trajectory", self.traj_path)]:
            if not path.exists():
                print(f"❌ Missing {name}: {path}")
                return None
        
        # 1. Load calibrations
        print("1️⃣ Loading calibrations...")
        self.load_calibrations()
        
        # 2. Load trajectory for reference frame
        print("2️⃣ Loading trajectory data...")
        df = pd.read_csv(self.traj_path)
        print(f"   Loaded {len(df)} trajectory frames")
        
        # 3. Establish CPF-aligned world frame
        print("3️⃣ Establishing CPF-aligned world frame...")
        self.establish_cpf_world_frame(df)
        
        # 4. Load and process IMU data
        print("4️⃣ Loading and processing IMU data...")
        imu_data = self._load_and_process_imu()
        
        # 5. Match trajectory to IMU timestamps
        print("5️⃣ Matching trajectory to IMU timestamps...")
        matched_data = self._match_trajectory_to_imu(df, imu_data['timestamps'])
        
        # Load Body
        body_data = self._load_body(imu_data['timecodes'])
        matched_body_data = self._match_body_to_imu(body_data)
        imu_data, matched_data, matched_body_data = self._adjust_all_data_times(imu_data, matched_data, matched_body_data)
        
        # 6. Transform all data
        print("6️⃣ Transforming data...")
        transformed_data = self._transform_to_cpf_world(imu_data, matched_data, matched_body_data)
        
        # 7. Resample to different frequencies
        print(f"7️⃣ Resampling: IMU to {imu_hz} Hz, Pose to {pose_hz} Hz...")
        final_data = self._resample_data_dual_rate(transformed_data, imu_hz, pose_hz)
        
        # 8. Validate velocity integration
        print("8️⃣ Validating velocity integration...")
        validation_results = self._validate_integration(final_data)
        final_data['validation'] = validation_results
        
        return final_data
    
    def _load_body(self, imu_timecodes):
        """load nymeria human body info"""
        import dataclasses

        from nymeria.data_provider import NymeriaDataProvider, NymeriaDataProviderConfig

        # Only pass options the installed loader actually declares, so this works both
        # with the upstream loader and with builds that add extra options.
        accepted = {f.name for f in dataclasses.fields(NymeriaDataProviderConfig)}
        kwargs = {"sequence_rootdir": self.sequence_path}
        for opt in ("load_wrist", "load_observer", "load_bbox"):
            if opt in accepted:
                kwargs[opt] = False

        # With the standard Nymeria layout the loader finds body motion at
        # <sequence_rootdir>/body/. If head and body were downloaded into separate
        # trees, point the loader at the body tree when it supports doing so.
        if "body_dir" in accepted:
            body_dir = Path(str(self.sequence_path).replace("head", "body_motion"))
            if body_dir.is_dir():
                kwargs["body_dir"] = str(body_dir)

        nymeria_dp = NymeriaDataProvider(**kwargs)
        timespan_ns = nymeria_dp.timespan_ns
        imu_dp = self.provider
        
        time_start_ns = imu_dp.convert_from_timecode_to_device_time_ns(int(timespan_ns[0]))
        time_end_ns = imu_dp.convert_from_timecode_to_device_time_ns(int(timespan_ns[1]))
        timestamps_timecode = imu_timecodes
        body_poses = []
        time = []
        mask = (timestamps_timecode >= timespan_ns[0]) & (timestamps_timecode <= timespan_ns[1])
        timestamps_timecode = timestamps_timecode[mask]

        for timecode in tqdm(timestamps_timecode, total=len(timestamps_timecode), desc="Processing timecodes"):
            data = nymeria_dp.get_synced_poses(timecode)
            body_tf = []
            for i in range(len(data['tf_xsens'])):
                body_tf.append(data['tf_xsens'][i].to_matrix())
            body_poses.append(np.array(body_tf))
            time.append(imu_dp.convert_from_timecode_to_device_time_ns(int(timecode)) * 1e-9)

        self.reference_timecode = imu_dp.convert_from_device_time_to_timecode_ns(int(self.reference_time*1e9))
        reference_data = nymeria_dp.get_synced_poses(self.reference_timecode)
        reference_body_tf = []
        for i in range(len(reference_data['tf_xsens'])):
            reference_body_tf.append(reference_data['tf_xsens'][i].to_matrix())
        self.reference_xsene_pose = np.array(reference_body_tf)
        
        return {
            'time_body': np.array(time),  # Start from 0
            'xsens_poses': np.array(body_poses),
            'time_start_ns': time_start_ns,
            'time_end_ns': time_end_ns
        }   
        
    def _load_and_process_imu(self):
        """Load and calibrate IMU data with correct transformation"""
        imu_id = self.provider.get_stream_id_from_label("imu-right")
        n_samples = self.provider.get_num_data(imu_id)
        
        # Find start index after skip period
        start_idx = 0
        if self.skip_seconds > 0:
            first_sample = self.provider.get_imu_data_by_index(imu_id, 0)
            target_time = first_sample.capture_timestamp_ns * 1e-9 + self.skip_seconds
            
            for i in range(n_samples):
                sample = self.provider.get_imu_data_by_index(imu_id, i)
                if sample.capture_timestamp_ns * 1e-9 >= target_time:
                    start_idx = i
                    break
        
        timestamps = []
        timecodes = []
        accels_cpf = []
        gyros_cpf = []
        
        print(f"   Processing {n_samples - start_idx} IMU samples...")
        for i in tqdm(range(start_idx, n_samples), desc="IMU", leave=False):
            sample = self.provider.get_imu_data_by_index(imu_id, i)
            
            # Apply calibration
            accel_raw = np.array(sample.accel_msec2)
            gyro_raw = np.array(sample.gyro_radsec)
            
            if self.imu_calib:
                accel_device = self.imu_calib['R_Device_Imu'] @ (accel_raw - self.imu_calib['accel_bias'])
                gyro_device = self.imu_calib['R_Device_Imu'] @ (gyro_raw - self.imu_calib['gyro_bias'])
            else:
                accel_device = accel_raw
                gyro_device = gyro_raw
            
            # CORRECTED: Transform to CPF using vector transformation
            accel_cpf = self.R_device_to_CPF_vectors @ accel_device
            gyro_cpf = self.R_device_to_CPF_vectors @ gyro_device
            timestamps.append(sample.capture_timestamp_ns * 1e-9)
            timecodes.append(self.provider.convert_from_device_time_to_timecode_ns(int(sample.capture_timestamp_ns)))
            accels_cpf.append(accel_cpf)
            gyros_cpf.append(gyro_cpf)
        
        # Verify gravity in first sample
        if len(accels_cpf) > 0:
            first_acc = accels_cpf[0]
            print(f"   First accel in CPF: [{first_acc[0]:.2f}, {first_acc[1]:.2f}, {first_acc[2]:.2f}]")
            print(f"   |accel|: {np.linalg.norm(first_acc):.2f} m/s² (should be ~9.8)")
        
        return {
            'timestamps': np.array(timestamps) - timestamps[0],  # Start from 0
            'timestamps_abs': np.array(timestamps),
            'timecodes': np.array(timecodes),
            'accelerometer_cpf': np.array(accels_cpf),
            'gyroscope_cpf': np.array(gyros_cpf)
        }
    
    def _adjust_all_data_times(self, imu_data, matched_data, matched_body_data):
        """Trim to overlap; align body to IMU by nearest timestamp (no interpolation)."""

        # --- time arrays ---
        t_body = np.asarray(matched_body_data['time_body'])
        if t_body.ndim != 1 or t_body.size == 0:
            raise ValueError("matched_body_data['time_body'] must be a non-empty 1D array")

        if 'timestamps_abs' not in imu_data:
            raise KeyError("imu_data must contain 'timestamps_abs'")
        t_imu = np.asarray(imu_data['timestamps_abs'])

        # --- overlap ---
        t0 = max(t_imu[0], t_body[0])
        t1 = min(t_imu[-1], t_body[-1])
        if t1 < t0:
            raise ValueError("No temporal overlap between IMU and body data")

        mask_imu  = (t_imu >= t0) & (t_imu <= t1)
        mask_body = (t_body >= t0) & (t_body <= t1)

        # --- slice imu & matched to overlap (unchanged lengths relative to masked imu) ---
        def _mask_dict_by_len(d, mask, expected_len):
            out = {}
            for k, v in d.items():
                arr = np.asarray(v)
                if arr.shape[:1] == (expected_len,):
                    out[k] = arr[mask]
                else:
                    out[k] = v
            return out

        imu_cut     = _mask_dict_by_len(imu_data,  mask_imu,  len(t_imu))
        matched_cut = _mask_dict_by_len(matched_data, mask_imu, len(t_imu))

        # --- body: nearest-neighbor to IMU times ---
        # source (overlap) times
        t_src_body = t_body[mask_body]
        t_tgt_imu  = t_imu[mask_imu]  # we want body samples at these times (by nearest)

        # find nearest indices in t_src_body for each t_tgt_imu
        # searchsorted assumes t_src_body is sorted (usual for timestamps)
        idx_right = np.searchsorted(t_src_body, t_tgt_imu, side='left')
        idx_right = np.clip(idx_right, 0, len(t_src_body) - 1)
        idx_left  = np.clip(idx_right - 1, 0, len(t_src_body) - 1)
        choose_right = np.abs(t_src_body[idx_right] - t_tgt_imu) <= np.abs(t_src_body[idx_left] - t_tgt_imu)
        nn_idx = np.where(choose_right, idx_right, idx_left)  # shape [len(t_tgt_imu)]

        # slice body dict by nearest indices for any field aligned on time_body
        body_overlap = _mask_dict_by_len(matched_body_data, mask_body, len(t_body))
        body_cut = {}
        for k, v in body_overlap.items():
            arr = np.asarray(v)
            if arr.shape[:1] == (len(t_src_body),):
                body_cut[k] = arr[nn_idx]
            else:
                body_cut[k] = v

        # Make body_cut time vector match IMU cut (optional but handy for 1:1 indexing)
        body_cut['time_body'] = t_tgt_imu.astype(np.float64)
        # If you’d like to keep the actual source times picked, also expose them:
        body_cut['time_body_src'] = t_src_body[nn_idx].astype(np.float64)

        return imu_cut, matched_cut, body_cut

        
        
    def _match_body_to_imu(self, body_data):
        """Match body data to IMU timestamps"""
        
        xsens_poses = body_data['xsens_poses']
        time_body = body_data['time_body']
        
        return {
            'xsens_poses': xsens_poses,
            'time_body': time_body
        }
        
    def _match_trajectory_to_imu(self, df, imu_timestamps):
        """Match trajectory data to IMU timestamps"""
        # Adjust for reference time
        traj_times = df['tracking_timestamp_us'].values / 1e6 - self.reference_time
        
        # Binary search for closest matches
        indices = np.searchsorted(traj_times, imu_timestamps)
        matched_indices = []
        
        for i, idx in enumerate(indices):
            if idx == 0:
                matched_idx = 0
            elif idx >= len(traj_times):
                matched_idx = len(traj_times) - 1
            else:
                # Find closest
                if (imu_timestamps[i] - traj_times[idx-1]) < (traj_times[idx] - imu_timestamps[i]):
                    matched_idx = idx - 1
                else:
                    matched_idx = idx
            matched_indices.append(matched_idx)
        
        matched_indices = np.array(matched_indices)
        
        # Extract matched data
        quaternions = np.array(df.iloc[matched_indices][['qw_world_device', 'qx_world_device', 
                                                         'qy_world_device', 'qz_world_device']].values, dtype=np.float64)
        positions = np.array(df.iloc[matched_indices][['tx_world_device', 'ty_world_device', 
                                                       'tz_world_device']].values, dtype=np.float64)
        
        # Calculate velocities from position derivatives if not available
        velocities = np.zeros((len(matched_indices), 3))
        if 'device_linear_velocity_x_device' in df.columns:
            velocities = np.array(df.iloc[matched_indices][['device_linear_velocity_x_device', 
                                                            'device_linear_velocity_y_device', 
                                                            'device_linear_velocity_z_device']].values, dtype=np.float64)
        else:
            # Compute from finite differences
            for i in range(1, len(matched_indices) - 1):
                dt = (traj_times[matched_indices[i+1]] - traj_times[matched_indices[i-1]])
                if dt > 0:
                    pos_next = positions[i+1]
                    pos_prev = positions[i-1]
                    vel_world = (pos_next - pos_prev) / dt
                    
                    # Transform to device frame
                    quat = quaternions[i]
                    R_device_to_world = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()
                    vel_device = R_device_to_world.T @ vel_world
                    velocities[i] = vel_device
        
        return {
            'quaternions': quaternions,
            'positions': positions,
            'velocities_device': velocities
        }
    
    def _transform_to_cpf_world(self, imu_data, matched_data, matched_body_data):
        """Transform all data with corrected transformation chain"""
        n_samples = len(imu_data['timestamps'])
        
        # Prepare output arrays
        orientations_cpfworld = []
        positions_cpfworld = []
        velocities_cpf = []
        xsens_poses_cpfworld = []
        
        print("   Transforming data...")
        print("   Saving velocities in CPF (body) frame for ML training")
        
        for i in tqdm(range(n_samples), desc="Transform", leave=False):
            # CORRECTED: Get device orientation (NO transpose)
            quat = matched_data['quaternions'][i]
            R_device_to_world = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()
            
            # CORRECTED: Transform to CPF using frame transformation
            R_CPF_to_world = R_device_to_world @ self.R_device_to_CPF_frames
            
            # Transform to CPF-aligned world
            R_CPF_to_cpfworld = self.R_world_to_cpfworld.T @ R_CPF_to_world
            
            # Store as quaternion [qx, qy, qz, qw] for PyTorch
            quat_cpfworld = Rotation.from_matrix(R_CPF_to_cpfworld).as_quat()
            orientations_cpfworld.append(quat_cpfworld)
            
            # Transform position to CPF-world
            pos_world = matched_data['positions'][i]
            relative_pos = pos_world - self.reference_position
            pos_cpfworld = self.R_world_to_cpfworld.T @ relative_pos
            positions_cpfworld.append(pos_cpfworld)
            
            # CORRECTED: Transform velocity to CPF using vector transformation
            vel_device = matched_data['velocities_device'][i]
            vel_cpf = self.R_device_to_CPF_vectors @ vel_device
            velocities_cpf.append(vel_cpf)
            
            # Transform body to CPF-world
            xsens_poses = matched_body_data['xsens_poses'][i]   # shape (J, 4, 4), world->body (T_wb)

            # Build world->cpfworld homogeneous transform once
            R_cw = self.R_world_to_cpfworld.T                   # 3x3
            t_cw = - R_cw @ self.reference_position             # 3,
            T_cw = np.eye(4)
            T_cw[:3, :3] = R_cw
            T_cw[:3, 3]  = t_cw

            # Apply change of frame: T_cb = T_cw @ T_wb  (broadcasted left-multiply over joints)
            xsens_poses_cpfworld.append((T_cw @ xsens_poses))        # (J, 4, 4)
        
        return {
            'timestamps': imu_data['timestamps'],
            'accelerometer_cpf': imu_data['accelerometer_cpf'],
            'gyroscope_cpf': imu_data['gyroscope_cpf'],
            'orientation_cpfworld': np.array(orientations_cpfworld, dtype=np.float32),
            'position_cpfworld': np.array(positions_cpfworld, dtype=np.float32),
            'xsens_pose_cpfworld': np.array(xsens_poses_cpfworld, dtype=np.float32),
            'velocity_cpf': np.array(velocities_cpf, dtype=np.float32),
        }
    
    def _resample_data_dual_rate(self, data, imu_hz, pose_hz):
        """Resample IMU data to high rate and pose data to lower rate
        
        Args:
            data: Input data dict
            imu_hz: Target Hz for IMU data (accel, gyro)
            pose_hz: Target Hz for pose data (orientation, position, velocity)
        """
        t_start = data['timestamps'][0]
        t_end = data['timestamps'][-1]
        
        # Create two time grids
        time_imu = np.arange(t_start, t_end, 1.0/imu_hz)
        time_pose = np.arange(t_start, t_end, 1.0/pose_hz)
        
        n_imu_samples = len(time_imu)
        n_pose_samples = len(time_pose)
        
        print(f"   Resampling from {len(data['timestamps'])} samples:")
        print(f"      IMU: {n_imu_samples} samples @ {imu_hz} Hz")
        print(f"      Pose: {n_pose_samples} samples @ {pose_hz} Hz")
        
        resampled = {}
        
        # Resample IMU data to high rate
        for key in ['accelerometer_cpf', 'gyroscope_cpf']:
            if key in data:
                source_data = np.array(data[key], dtype=np.float32)
                arr = np.zeros((n_imu_samples, 3), dtype=np.float32)
                for i in range(3):
                    arr[:, i] = np.interp(time_imu, data['timestamps'], source_data[:, i])
                resampled[key] = arr
        
        # Resample pose data to lower rate
        for key in ['position_cpfworld', 'velocity_cpf']:
            if key in data:
                source_data = np.array(data[key], dtype=np.float32)
                arr = np.zeros((n_pose_samples, 3), dtype=np.float32)
                for i in range(3):
                    arr[:, i] = np.interp(time_pose, data['timestamps'], source_data[:, i])
                resampled[key] = arr
        
        # Resample quaternions (4D) to lower rate
        if 'orientation_cpfworld' in data:
            source_data = np.array(data['orientation_cpfworld'], dtype=np.float32)
            arr = np.zeros((n_pose_samples, 4), dtype=np.float32)
            
            # Simple linear interpolation (could use SLERP for better results)
            for i in range(4):
                arr[:, i] = np.interp(time_pose, data['timestamps'], source_data[:, i])
            
            # Normalize quaternions
            arr = arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-8)
            resampled['orientation_cpfworld'] = arr
            
        if 'xsens_pose_cpfworld' in data:
            ts = np.asarray(data['timestamps'], dtype=np.float64)
            T_src = np.asarray(data['xsens_pose_cpfworld'], dtype=np.float64)    # [N, J, 4, 4]
            N, J = T_src.shape[0], T_src.shape[1]

            R_src = T_src[..., :3, :3]                                           # [N, J, 3, 3]
            p_src = T_src[..., :3, 3]                                            # [N, J, 3]

            # Interp translations per joint
            p_out = np.empty((n_pose_samples, J, 3), dtype=np.float64)
            for j in range(J):
                for k in range(3):
                    p_out[:, j, k] = np.interp(time_pose, ts, p_src[:, j, k])

            # SLERP rotations per joint
            R_out = np.empty((n_pose_samples, J, 3, 3), dtype=np.float64)
            # Convert all Rs for joint j to Rotations once per joint to keep continuity
            for j in range(J):
                rj_src = Rotation.from_matrix(R_src[:, j, :, :])                 # length N
                slerp_j = Slerp(ts, rj_src)
                R_out[:, j, :, :] = slerp_j(time_pose).as_matrix()

            # Rebuild homogeneous transforms
            T_out = np.tile(np.eye(4, dtype=np.float64), (n_pose_samples, J, 1, 1))
            T_out[..., :3, :3] = R_out
            T_out[..., :3, 3]  = p_out
            resampled['xsens_pose_cpfworld'] = T_out.astype(np.float32)
        
        # Store both time grids and sampling rates
        resampled['timestamps_imu'] = time_imu.astype(np.float32)
        resampled['timestamps_pose'] = time_pose.astype(np.float32)
        resampled['imu_sampling_rate_hz'] = imu_hz
        resampled['pose_sampling_rate_hz'] = pose_hz
        resampled['num_imu_samples'] = n_imu_samples
        resampled['num_pose_samples'] = n_pose_samples
        
        return resampled
    
    def _validate_integration(self, data):
        """Validate velocity integration with corrected transformation"""
        # Use pose timestamps for validation
        timestamps = data['timestamps_pose']
        n_samples = len(timestamps)
        
        print("\\n   🔍 VALIDATION: Integrating CPF (body frame) velocities")
        
        # Integrate velocities
        integrated_cpfworld = np.zeros_like(data['position_cpfworld'])
        integrated_cpfworld[0] = data['position_cpfworld'][0]
        
        for i in range(1, n_samples):
            dt = timestamps[i] - timestamps[i-1]
            
            # Get CPF to CPF-world rotation
            quat_cpfworld = data['orientation_cpfworld'][i-1]
            R_CPF_to_cpfworld = Rotation.from_quat(quat_cpfworld).as_matrix()
            
            # CORRECTED: Transform CPF velocity to CPF-world (no transpose!)
            vel_cpf = data['velocity_cpf'][i-1]
            vel_cpfworld = R_CPF_to_cpfworld @ vel_cpf
            
            # Integrate in CPF-world frame
            integrated_cpfworld[i] = integrated_cpfworld[i-1] + vel_cpfworld * dt
        
        errors_cpfworld = np.linalg.norm(integrated_cpfworld - data['position_cpfworld'], axis=1)
        
        # Calculate metrics
        total_distance = np.sum(np.linalg.norm(np.diff(data['position_cpfworld'], axis=0), axis=1))
        
        validation = {
            'max_error': float(np.max(errors_cpfworld)),
            'mean_error': float(np.mean(errors_cpfworld)),
            'final_error': float(errors_cpfworld[-1]),
            'drift_percentage': float((errors_cpfworld[-1] / total_distance * 100)) if total_distance > 0 else 0.0,
            'total_distance': float(total_distance),
            'duration': float(timestamps[-1] - timestamps[0])
        }
        
        print(f"\\n   📊 Integration Validation Results:")
        print(f"      Max error: {validation['max_error']:.3f} m")
        print(f"      Mean error: {validation['mean_error']:.3f} m")
        print(f"      Final drift: {validation['drift_percentage']:.2f}%")
        print(f"      Total distance: {validation['total_distance']:.1f} m")
        
        return validation


def process_sequence_wrapper(args):
    """Wrapper for parallel processing"""
    sequence_path, output_dir, skip_seconds, imu_hz, pose_hz = args
    
    
    output_path = output_dir / f"{sequence_path.name}_cpfbody_imu{imu_hz}hz_pose{pose_hz}hz.pkl"
    if output_path.exists():
        print("File already exists:", output_path)
        return None

    try:
        transformer = CPFWorldTransformer(sequence_path, skip_seconds)
        data = transformer.process_full_sequence(imu_hz, pose_hz)
        
        if data is None:
            return None
        
        # Prepare output
        output_data = {
            # Metadata
            'sequence_name': sequence_path.name,
            'duration_seconds': float(data['timestamps_pose'][-1] - data['timestamps_pose'][0]),
            'num_imu_samples': int(data['num_imu_samples']),
            'num_pose_samples': int(data['num_pose_samples']),
            'imu_sampling_rate_hz': float(data['imu_sampling_rate_hz']),
            'pose_sampling_rate_hz': float(data['pose_sampling_rate_hz']),
            
            # Frame info
            'coordinate_frames': {
                'imu': 'CPF (body frame)',
                'velocity': 'CPF (body frame)',
                'orientation': 'CPF→CPF-world quaternions',
                'position': 'CPF-world frame'
            },
            'reference_time': float(transformer.reference_time),
            'reference_position': transformer.reference_position.astype(np.float32),
            
            # Transformation matrices
            'R_device_to_CPF_frames': transformer.R_device_to_CPF_frames.astype(np.float32),
            'R_device_to_CPF_vectors': transformer.R_device_to_CPF_vectors.astype(np.float32),
            'R_world_to_cpfworld': transformer.R_world_to_cpfworld.astype(np.float32),
            
            # Time-series data as torch tensors with different sampling rates
            'imu_data': {
                'time': torch.tensor(data['timestamps_imu']),
                'timestamps_imu': torch.tensor(data['timestamps_imu']),  # For dual-rate detection
                'accel': torch.tensor(data['accelerometer_cpf']),
                'gyro': torch.tensor(data['gyroscope_cpf']),
            },
            'gt_data': {
                'time': torch.tensor(data['timestamps_pose']),
                'timestamps_pose': torch.tensor(data['timestamps_pose']),  # For dual-rate detection
                'orientation': torch.tensor(data['orientation_cpfworld']),
                'position': torch.tensor(data['position_cpfworld']),
                'velocity': torch.tensor(data['velocity_cpf']),
                'xsens_pose': torch.tensor(data['xsens_pose_cpfworld']),
            },
            'validation': data['validation']
        }
        
        # Save
        output_path = output_dir / f"{sequence_path.name}_cpfbody_imu{imu_hz}hz_pose{pose_hz}hz.pkl"
        with open(output_path, 'wb') as f:
            pickle.dump(output_data, f)
        
        print(f"✅ Saved: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"❌ Failed to process {sequence_path.name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description="Process Nymeria sequences with corrected CPF transformations")
    parser.add_argument("path", help="Path to Nymeria sequence(s)")
    parser.add_argument("-o", "--output-dir", default="output_final", help="Output directory")
    parser.add_argument("--max", type=int, help="Max sequences to process")
    parser.add_argument("--skip-seconds", type=int, default=30, help="Seconds to skip at start")
    parser.add_argument("--imu-hz", type=int, default=50, help="Target sampling rate for IMU data in Hz")
    parser.add_argument("--pose-hz", type=int, default=50, help="Target sampling rate for pose data in Hz")
    parser.add_argument("--parallel", type=int, default=16, help="Number of parallel workers")
    parser.add_argument("--specificlist", help="Process a list of specific sequence by name")
    args = parser.parse_args()
    
    input_path = Path(args.path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("="*80)
    print("🚀 CORRECTED CPF TRANSFORMATION PIPELINE")
    print("="*80)
    print(f"📂 Input: {input_path}")
    print(f"📂 Output: {output_dir}")
    print(f"⏰ IMU Hz: {args.imu_hz}, Pose Hz: {args.pose_hz}")
    print(f"⏭️  Skip seconds: {args.skip_seconds}")
    
    # Find sequences
    sequences = []
    if (input_path / "recording_head" / "data" / "motion.vrs").exists():
        sequences = [input_path]
    else:
        sequences = sorted([d for d in input_path.iterdir() if d.is_dir() and not d.name.startswith('.')])
    
    if args.max:
        sequences = sequences[:args.max]
    if args.specificlist:
        try:
            with open(args.specificlist, 'r') as f:
                specific_names = [line.strip() for line in f if line.strip()]
            matching_sequences = [seq for seq in sequences if any(name in seq.name for name in specific_names)]
            if not matching_sequences:
                print(f"No sequences from list in '{args.specificlist}' matched. Exiting.")
                exit(1)
            sequences = matching_sequences
            print(f"Selected {len(sequences)} sequences from list in '{args.specificlist}'")
        except FileNotFoundError:
            print(f"Specific list file '{args.specificlist}' not found. Exiting.")
            exit(1)
            
    print(f"📁 Found {len(sequences)} sequences")
    
    # Process
    if len(sequences) == 1:
        # Single sequence
        result = process_sequence_wrapper((sequences[0], output_dir, args.skip_seconds, args.imu_hz, args.pose_hz))
        success_count = 1 if result else 0
        failed_sequences = [] if result else [sequences[0].name]
    else:
        # Parallel processing
        print(f"🔧 Processing with {args.parallel} workers...")
        process_args = [(seq, output_dir, args.skip_seconds, args.imu_hz, args.pose_hz) for seq in sequences]
        
        with Pool(args.parallel) as pool:
            results = list(tqdm(
                pool.imap(process_sequence_wrapper, process_args),
                total=len(sequences),
                desc="Processing"
            ))
        
        success_count = sum(1 for r in results if r is not None)
        failed_sequences = [sequences[i].name for i, r in enumerate(results) if r is None]
    
    # Summary
    print(f"\\n{'='*80}")
    print(f"✅ PROCESSING COMPLETE")
    print(f"{'='*80}")
    print(f"Successful: {success_count}/{len(sequences)}")
    if failed_sequences:
        print(f"Failed: {', '.join(failed_sequences)}")
    
    print(f"\\n📋 Corrected Transformations Applied:")
    print(f"   1. SLAM quaternion: NO transpose")
    print(f"   2. Device→CPF frames: R_raw (no transpose)")
    print(f"   3. Device→CPF vectors: R_raw.T (with transpose)")
    print(f"   4. Velocity integration: vel_world = R_cpf_to_world @ vel_cpf")
    print(f"\\n📁 Output: {output_dir}")

if __name__ == "__main__":
    main()
    
    
# python preprocess_nymeria_body.py $DATA_ROOT/nymeria_head_full -o $DATA_ROOT/processed_data_body_Ty --imu-hz 50 --pose-hz 50 