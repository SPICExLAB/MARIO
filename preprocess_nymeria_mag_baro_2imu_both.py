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
        
        # Load IMU calibrations (separate for left and right)
        self.imu_calib_right, self.imu_calib_left = self._load_imu_calibrations()

        print(f"✅ Loaded calibrations")
        print(f"   R_device_to_CPF_frames det: {np.linalg.det(self.R_device_to_CPF_frames):.6f}")
        print(f"   Using R_raw for frames, R_raw.T for vectors")

    def _load_imu_calibrations(self):
        """Load IMU calibrations from file (both left and right)"""
        if not self.calib_path.exists():
            print("⚠️ No IMU calibration found, using identity")
            return None, None

        calibrations = []
        with open(self.calib_path, 'r') as f:
            for line in f:
                calibrations.append(json.loads(line))

        imu_right = None
        imu_left = None

        for calib in calibrations:
            for imu_calib in calib.get('ImuCalibrations', []):
                T_Device_Imu = imu_calib['T_Device_Imu']
                quat = T_Device_Imu['UnitQuaternion']
                qw = quat[0]
                qx, qy, qz = quat[1][0], quat[1][1], quat[1][2]
                R_Device_Imu = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()

                calib_dict = {
                    'R_Device_Imu': R_Device_Imu,
                    'gyro_bias': np.array(imu_calib['Gyroscope']['Bias']['Offset']),
                    'accel_bias': np.array(imu_calib['Accelerometer']['Bias']['Offset']),
                }

                if imu_calib['Label'] == 'imu-right':
                    imu_right = calib_dict
                elif imu_calib['Label'] == 'imu-left':
                    imu_left = calib_dict

        print(f"   IMU-right calibration: {'✓' if imu_right else '✗'}")
        print(f"   IMU-left calibration:  {'✓' if imu_left else '✗'}")

        return imu_right, imu_left
    
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
        
    def process_full_sequence(self, sample_hz=200):
        """Process entire sequence with corrected transformations

        Args:
            sample_hz: Sampling rate for all data (default 200Hz)
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
        
        # Load Mag/Baro
        mag_baro_data = self._load_mag_baro()
        matched_mag_baro_data = self._match_baro_mag_data(mag_baro_data,imu_data['timestamps_abs'])
        
        # Adjust all data
        imu_data, matched_data, matched_mag_baro_data = self._adjust_all_data_times(imu_data, matched_data, matched_mag_baro_data)
        
        # 6. Transform all data
        print("6️⃣ Transforming data...")
        transformed_data = self._transform_to_cpf_world(imu_data, matched_data, matched_mag_baro_data)

        # 7. Resample to single frequency
        print(f"7️⃣ Resampling to {sample_hz} Hz...")
        final_data = self._resample_data_single_rate(transformed_data, sample_hz)

        # 8. Validate velocity integration
        print("8️⃣ Validating velocity integration...")
        validation_results = self._validate_integration(final_data)
        final_data['validation'] = validation_results

        return final_data
    
    def _load_mag_baro(self,):
        """load baro and mag"""

        provider = data_provider.create_vrs_data_provider(str(self.vrs_path))
        
        if not provider:
            raise ValueError(f"Failed to create Data VRS provider for {self.vrs_path}")
        
        # Get Mag stream ID
        mag_stream_id = provider.get_stream_id_from_label("mag0")
        mag_calib = provider.get_sensor_calibration(mag_stream_id)
        magnetometer_calib = mag_calib.magnetometer_calibration()
        
        if not mag_stream_id:
            raise ValueError("Mag stream not found in Data VRS file")
        
        print(f"Found mag stream ID: {mag_stream_id}")
        mag_num_samples = provider.get_num_data(mag_stream_id)
        print(f"Total mag samples: {mag_num_samples}")
        
        # Extract data
        mag_x = []
        mag_y = []
        mag_z = []
        mag_cali_ls = []
        mag_timestamps = []

        for index in range(0, mag_num_samples):
            mag_data = provider.get_magnetometer_data_by_index(mag_stream_id, index)
            raw = np.array([[mag_data.mag_tesla[0]],
                            [mag_data.mag_tesla[1]],
                            [mag_data.mag_tesla[2]]], dtype=np.float64)
            mag_cali = magnetometer_calib.raw_to_rectified(raw)
            mag_cali_ls.append(mag_cali * 1e6)
            mx = mag_data.mag_tesla[0] * 1e6
            my = mag_data.mag_tesla[1] * 1e6
            mz = mag_data.mag_tesla[2] * 1e6
            mag_x.append(mx)
            mag_y.append(my)
            mag_z.append(mz)
            mag_timestamps.append(mag_data.capture_timestamp_ns * 1e-9)
            
        # Get Baro stream ID
        baro_stream_id = provider.get_stream_id_from_label("baro0")
        baro_calib = provider.get_sensor_calibration(baro_stream_id)
        barometer_calib = baro_calib.barometer_calibration()
        
        if not baro_stream_id:
            raise ValueError("Baro stream not found in Data VRS file")
        
        print(f"Found baro stream ID: {baro_stream_id}")
        baro_num_samples = provider.get_num_data(baro_stream_id)
        print(f"Total baro samples: {baro_num_samples}")
        
        # Extract data from IMU
        pressure = []
        temperature = []
        altitude = []
        pressure_cali_ls = []
        altitude_cali_ls = []
        baro_timestamps = []

        P0 = 101325  # Standard sea-level pressure in Pa

        for index in range(0, baro_num_samples):
            baro_data = provider.get_barometer_data_by_index(baro_stream_id, index)
            raw = baro_data.pressure
            p_pa_cali = barometer_calib.raw_to_rectified(raw)
            
            p_kpa = baro_data.pressure * 1e-3
            p_pa = baro_data.pressure  # Already in Pascals
            pressure.append(p_kpa)
            temperature.append(baro_data.temperature)
            baro_timestamps.append(baro_data.capture_timestamp_ns * 1e-9)

            # Calculate altitude
            h = 44330 * (1 - (p_pa / P0) ** (1 / 5.255))
            altitude.append(h)
            
            pressure_cali_ls.append(p_pa_cali * 1e-3)
            h_cali = 44330 * (1 - (p_pa_cali / P0) ** (1 / 5.255))
            altitude_cali_ls.append(h_cali)

        result = {
            "mag_x": np.array(mag_x, dtype=np.float32),
            "mag_y": np.array(mag_y, dtype=np.float32),
            "mag_z": np.array(mag_z, dtype=np.float32),
            "mag_cali": np.array(mag_cali_ls, dtype=np.float32),
            "mag_timestamps": np.array(mag_timestamps, dtype=np.float32),
            "pressure": np.array(pressure, dtype=np.float32), 
            "temperature": np.array(temperature, dtype=np.float32),
            "altitude": np.array(altitude, dtype=np.float32),
            "pressure_cali": np.array(pressure_cali_ls, dtype=np.float32),
            "altitude_cali": np.array(altitude_cali_ls, dtype=np.float32),
            "baro_timestamps": np.array(baro_timestamps, dtype=np.float32)
        }
        
        return result

        
    def _match_baro_mag_data(self, mag_baro_data, imu_data_timestamps_abs):
        """
        Align/interpolate magnetometer & barometer data to IMU absolute timestamps.

        Args:
            mag_baro_data (dict): Output from _load_baro_mag()
                (expects 'mag_timestamps', 'baro_timestamps', and associated arrays).
            imu_data_timestamps_abs (np.ndarray): Absolute IMU timestamps in seconds.

        Returns:
            dict aligned to imu_data_timestamps_abs, or None if no streams available.
        """
        import numpy as np

        if mag_baro_data is None or imu_data_timestamps_abs is None or len(imu_data_timestamps_abs) == 0:
            return None

        t_imu = np.asarray(imu_data_timestamps_abs, dtype=np.float64)
        N = t_imu.size

        out = {
            "time_mag_baro": t_imu.astype(np.float32),
            "mag": np.full((N, 3), np.nan, dtype=np.float32),
            "mag_cali": np.full((N, 3), np.nan, dtype=np.float32),
            "pressure": np.full(N, np.nan, dtype=np.float32),
            "pressure_cali": np.full(N, np.nan, dtype=np.float32),
            "temperature": np.full(N, np.nan, dtype=np.float32),
            "altitude": np.full(N, np.nan, dtype=np.float32),
            "altitude_cali": np.full(N, np.nan, dtype=np.float32),
        }

        # ---- Magnetometer (raw) ----
        t_mag = np.asarray(mag_baro_data.get("mag_timestamps", []), dtype=np.float64)
        if t_mag.size >= 2:  # need at least two points for interp
            mx = np.asarray(mag_baro_data.get("mag_x", []), dtype=np.float64)
            my = np.asarray(mag_baro_data.get("mag_y", []), dtype=np.float64)
            mz = np.asarray(mag_baro_data.get("mag_z", []), dtype=np.float64)
            if mx.size == t_mag.size and my.size == t_mag.size and mz.size == t_mag.size:
                out["mag"][:, 0] = np.interp(t_imu, t_mag, mx).astype(np.float32)
                out["mag"][:, 1] = np.interp(t_imu, t_mag, my).astype(np.float32)
                out["mag"][:, 2] = np.interp(t_imu, t_mag, mz).astype(np.float32)

            # Calibrated mag may be (N,3,1); squeeze to (N,3)
            mag_cali = mag_baro_data.get("mag_cali", None)
            if mag_cali is not None:
                mc = np.asarray(mag_cali, dtype=np.float64).squeeze()
                if mc.ndim == 2 and mc.shape == (t_mag.size, 3):
                    for i in range(3):
                        out["mag_cali"][:, i] = np.interp(t_imu, t_mag, mc[:, i]).astype(np.float32)

        # ---- Barometer ----
        t_baro = np.asarray(mag_baro_data.get("baro_timestamps", []), dtype=np.float64)
        if t_baro.size >= 2:
            def _interp_field(key, dest_key):
                vals = np.asarray(mag_baro_data.get(key, []), dtype=np.float64)
                if vals.size == t_baro.size:
                    out[dest_key][:] = np.interp(t_imu, t_baro, vals).astype(np.float32)

            _interp_field("pressure", "pressure")           # kPa
            _interp_field("temperature", "temperature")     # C
            _interp_field("altitude", "altitude")           # m
            _interp_field("pressure_cali", "pressure_cali") # kPa
            _interp_field("altitude_cali", "altitude_cali") # m

        has_any = (
            np.isfinite(out["mag"]).any()
            or np.isfinite(out["mag_cali"]).any()
            or np.isfinite(out["pressure"]).any()
            or np.isfinite(out["pressure_cali"]).any()
            or np.isfinite(out["temperature"]).any()
            or np.isfinite(out["altitude"]).any()
            or np.isfinite(out["altitude_cali"]).any()
        )
        return out if has_any else None
    
    def _load_and_process_imu(self):
        """Load, calibrate, and align right/left IMU data on a common timeline.

        - Right IMU ("imu-right"): applies optional calibration + rotation to CPF.
        - Left IMU ("imu-left"): raw readings, resampled (interpolated) to the
        overlap of the right IMU absolute timestamps.
        - Returns arrays all matched in length and time base (the overlap range).
        """
        import numpy as np
        from tqdm import tqdm

        # ---------- Right IMU: load, optionally skip initial seconds, calibrate, rotate ----------
        imu_id = self.provider.get_stream_id_from_label("imu-right")
        n_samples = self.provider.get_num_data(imu_id)

        # Find start index after skip period
        start_idx = 0
        if getattr(self, "skip_seconds", 0) > 0:
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

        print(f"   Processing {n_samples - start_idx} IMU samples (right)...")
        for i in tqdm(range(start_idx, n_samples), desc="IMU-right", leave=False):
            sample = self.provider.get_imu_data_by_index(imu_id, i)

            # Raw right-IMU measurements
            accel_raw = np.array(sample.accel_msec2, dtype=np.float64)
            gyro_raw  = np.array(sample.gyro_radsec,  dtype=np.float64)

            # Apply calibration if provided
            if self.imu_calib_right:
                R = self.imu_calib_right['R_Device_Imu']
                accel_device = R @ (accel_raw - self.imu_calib_right['accel_bias'])
                gyro_device  = R @ (gyro_raw  - self.imu_calib_right['gyro_bias'])
            else:
                accel_device = accel_raw
                gyro_device  = gyro_raw
            breakpoint()
            # Transform to CPF (vector transform)
            accel_cpf = self.R_device_to_CPF_vectors @ accel_device
            gyro_cpf  = self.R_device_to_CPF_vectors @ gyro_device

            t_abs = sample.capture_timestamp_ns * 1e-9
            timestamps.append(t_abs)
            timecodes.append(self.provider.convert_from_device_time_to_timecode_ns(int(sample.capture_timestamp_ns)))
            accels_cpf.append(accel_cpf)
            gyros_cpf.append(gyro_cpf)

        # Verify gravity in first RIGHT-IMU sample (CPF)
        if len(accels_cpf) > 0:
            first_acc = accels_cpf[0]
            print(f"   First accel in CPF (right): [{first_acc[0]:.2f}, {first_acc[1]:.2f}, {first_acc[2]:.2f}]")
            print(f"   |accel|: {np.linalg.norm(first_acc):.2f} m/s² (should be ~9.8)")

        # Convert RIGHT lists to arrays
        tR = np.asarray(timestamps, dtype=np.float64)
        timecodes = np.asarray(timecodes)
        accels_cpf = np.asarray(accels_cpf, dtype=np.float64)  # (N_r,3)
        gyros_cpf  = np.asarray(gyros_cpf,  dtype=np.float64)  # (N_r,3)

        # ---------- Left IMU: load and transform to CPF (same as right IMU) ----------
        imu_id_left = self.provider.get_stream_id_from_label("imu-left")
        num_samples_left = self.provider.get_num_data(imu_id_left)

        timestamps_left = []
        timecodes_left = []
        accels_left_cpf = []
        gyros_left_cpf = []
        accels_left_body = []  # Left IMU in its own body frame
        gyros_left_body = []   # Left IMU in its own body frame

        print(f"   Processing {num_samples_left} IMU samples (left)...")
        for i in tqdm(range(num_samples_left), desc="IMU-left", leave=False):
            sample = self.provider.get_imu_data_by_index(imu_id_left, i)
            t_abs = sample.capture_timestamp_ns * 1e-9

            timestamps_left.append(t_abs)
            timecodes_left.append(self.provider.convert_from_device_time_to_timecode_ns(int(sample.capture_timestamp_ns)))

            # Raw left-IMU measurements
            accel_raw = np.array(sample.accel_msec2, dtype=np.float64)
            gyro_raw = np.array(sample.gyro_radsec, dtype=np.float64)

            # Apply LEFT IMU calibration if provided
            if self.imu_calib_left:
                R = self.imu_calib_left['R_Device_Imu']
                accel_device = R @ (accel_raw - self.imu_calib_left['accel_bias'])
                gyro_device = R @ (gyro_raw - self.imu_calib_left['gyro_bias'])
            else:
                accel_device = accel_raw
                gyro_device = gyro_raw

            # Store left IMU in its own body frame (after calibration, before CPF transform)
            accels_left_body.append(accel_device)
            gyros_left_body.append(gyro_device)

            # Transform to CPF (vector transform) - SAME AS RIGHT IMU
            accel_cpf = self.R_device_to_CPF_vectors @ accel_device
            gyro_cpf = self.R_device_to_CPF_vectors @ gyro_device

            accels_left_cpf.append(accel_cpf)
            gyros_left_cpf.append(gyro_cpf)

        # Convert LEFT lists to arrays
        tL = np.asarray(timestamps_left, dtype=np.float64)
        accL_cpf = np.asarray(accels_left_cpf, dtype=np.float64)    # (N_l,3) in CPF frame
        gyrL_cpf = np.asarray(gyros_left_cpf,  dtype=np.float64)    # (N_l,3) in CPF frame
        accL_body = np.asarray(accels_left_body, dtype=np.float64)  # (N_l,3) in left IMU body frame
        gyrL_body = np.asarray(gyros_left_body,  dtype=np.float64)  # (N_l,3) in left IMU body frame

        # ---------- Align LEFT to RIGHT on the overlap window ----------
        if tR.size == 0 or tL.size == 0:
            raise RuntimeError("Empty IMU stream(s): cannot align left/right.")

        # np.interp requires strictly increasing x; deduplicate equal timestamps on LEFT
        tL_u, idx_u = np.unique(tL, return_index=True)
        accL_cpf_u = accL_cpf[idx_u]
        gyrL_cpf_u = gyrL_cpf[idx_u]
        accL_body_u = accL_body[idx_u]
        gyrL_body_u = gyrL_body[idx_u]

        # Compute absolute-time overlap
        t_start = max(tR[0], tL_u[0])
        t_end   = min(tR[-1], tL_u[-1])

        mask_R = (tR >= t_start) & (tR <= t_end)
        if not np.any(mask_R):
            raise RuntimeError("No time overlap between left and right IMU streams.")

        # Crop RIGHT arrays to overlap; this becomes the common time base
        tR_sync = tR[mask_R]
        timecodes_sync   = timecodes[mask_R]
        accels_cpf_sync  = accels_cpf[mask_R].astype(np.float32)
        gyros_cpf_sync   = gyros_cpf[mask_R].astype(np.float32)

        # Interpolate LEFT accel/gyro to RIGHT's (cropped) absolute timestamps
        def interp_3d(t_src, V_src, t_dst):
            out = np.empty((t_dst.shape[0], 3), dtype=np.float64)
            for k in range(3):
                out[:, k] = np.interp(t_dst, t_src, V_src[:, k])
            return out

        accL_cpf_on_R   = interp_3d(tL_u, accL_cpf_u, tR_sync).astype(np.float32)
        gyrL_cpf_on_R   = interp_3d(tL_u, gyrL_cpf_u, tR_sync).astype(np.float32)
        accL_body_on_R  = interp_3d(tL_u, accL_body_u, tR_sync).astype(np.float32)
        gyrL_body_on_R  = interp_3d(tL_u, gyrL_body_u, tR_sync).astype(np.float32)

        # Verify both IMUs are in CPF frame (check first sample)
        if len(accels_cpf_sync) > 0:
            first_acc_right = accels_cpf_sync[0]
            print(f"   First accel in CPF (right): [{first_acc_right[0]:.2f}, {first_acc_right[1]:.2f}, {first_acc_right[2]:.2f}]")
            print(f"   |accel|: {np.linalg.norm(first_acc_right):.2f} m/s² (should be ~9.8)")
        if len(accL_cpf_on_R) > 0:
            first_acc_left = accL_cpf_on_R[0]
            print(f"   First accel in CPF (left):  [{first_acc_left[0]:.2f}, {first_acc_left[1]:.2f}, {first_acc_left[2]:.2f}]")
            print(f"   |accel|: {np.linalg.norm(first_acc_left):.2f} m/s² (should be ~9.8)")

        print(f"   ✅ Both IMUs aligned in CPF frame (dataloader will handle fusion)")

        # Zeroed relative timestamps (start from 0) using the common, cropped timeline
        timestamps_zeroed = (tR_sync - tR_sync[0]).astype(np.float64)

        # ---------- Return (all arrays now share the same length/time base) ----------
        return {
            # RIGHT timeline (cropped to overlap), starting at 0
            'timestamps': timestamps_zeroed,             # shape: (M,)
            'timestamps_abs': tR_sync,                   # shape: (M,)
            'timecodes': timecodes_sync,                 # shape: (M,)

            # Right IMU in CPF frame
            'accelerometer_cpf': accels_cpf_sync,        # shape: (M,3) - RIGHT in CPF
            'gyroscope_cpf': gyros_cpf_sync,             # shape: (M,3) - RIGHT in CPF

            # Left IMU in CPF frame (for ablation/fusion in dataloader)
            'accel_left_cpf': accL_cpf_on_R,             # shape: (M,3) - LEFT in CPF
            'gyro_left_cpf':  gyrL_cpf_on_R,             # shape: (M,3) - LEFT in CPF

            # Left IMU in its own body frame (for ablation studies)
            'accel_left_body': accL_body_on_R,           # shape: (M,3) - LEFT in own body frame
            'gyro_left_body':  gyrL_body_on_R,           # shape: (M,3) - LEFT in own body frame
        }

    def _adjust_all_data_times(
        self,
        imu_data,
        matched_data,
        matched_mag_baro_data,
        *,
        mb_time_key="time_mag_baro",   # change if your key differs
        imu_time_key="timestamps_abs",
    ):
        """Trim to overlap; align body & mag/baro streams to IMU by nearest timestamp (no interpolation)."""

        import numpy as np

        # --- time arrays ---
        if imu_time_key not in imu_data:
            raise KeyError(f"imu_data must contain '{imu_time_key}'")
        t_imu = np.asarray(imu_data[imu_time_key])

        if matched_mag_baro_data is None:
            raise ValueError("matched_mag_baro_data must be provided")
        if mb_time_key not in matched_mag_baro_data:
            raise KeyError(f"matched_mag_baro_data must contain '{mb_time_key}'")
        t_mb = np.asarray(matched_mag_baro_data[mb_time_key])
        if t_mb.ndim != 1 or t_mb.size == 0:
            raise ValueError(f"matched_mag_baro_data['{mb_time_key}'] must be a non-empty 1D array")

        # --- overlap across ALL THREE (IMU, body, mag/baro) ---
        t0 = max(t_imu[0], t_mb[0])
        t1 = min(t_imu[-1], t_mb[-1])
        if t1 < t0:
            raise ValueError("No temporal overlap among IMU, body, and mag/baro data")

        mask_imu  = (t_imu >= t0) & (t_imu <= t1)
        mask_mb   = (t_mb   >= t0) & (t_mb   <= t1)

        # --- helper to mask dicts by a boolean mask when first dim matches an expected length ---
        def _mask_dict_by_len(d, mask, expected_len):
            out = {}
            for k, v in d.items():
                arr = np.asarray(v)
                if arr.shape[:1] == (expected_len,):
                    out[k] = arr[mask]
                else:
                    out[k] = v
            return out

        # slice imu & matched to overlap (unchanged lengths relative to masked imu)
        imu_cut     = _mask_dict_by_len(imu_data,        mask_imu,  len(t_imu))
        matched_cut = _mask_dict_by_len(matched_data,    mask_imu,  len(t_imu))

        # --- nearest neighbor helper (align source timeline to target timeline) ---
        def _align_by_nearest(src_times, tgt_times, src_dict):
            # assumes src_times are sorted; if not sure, sort + reindex
            # (most timestamp arrays are sorted, but this is robust)
            order = np.argsort(src_times)
            src_times_sorted = src_times[order]

            # map from tgt_times -> nearest indices in src_times_sorted
            idx_right = np.searchsorted(src_times_sorted, tgt_times, side='left')
            idx_right = np.clip(idx_right, 0, len(src_times_sorted) - 1)
            idx_left  = np.clip(idx_right - 1, 0, len(src_times_sorted) - 1)
            choose_right = np.abs(src_times_sorted[idx_right] - tgt_times) <= np.abs(src_times_sorted[idx_left] - tgt_times)
            nn_sorted = np.where(choose_right, idx_right, idx_left)

            # convert to indices in original (unsorted) space
            nn_idx = order[nn_sorted]

            # gather any field whose first dimension matches len(src_times)
            out = {}
            for k, v in src_dict.items():
                arr = np.asarray(v)
                if arr.shape[:1] == (len(src_times),):
                    out[k] = arr[nn_idx]
                else:
                    out[k] = v

            # provide both the aligned target times (IMU cut) and the original picked source times
            out_time_key = next((k for k in (mb_time_key) if k in src_dict), None)
            if out_time_key is not None:
                out[out_time_key] = tgt_times.astype(np.float64)
                out[out_time_key + "_src"] = src_times[nn_idx].astype(np.float64)
            return out

        # times we align to (IMU within overlap)
        t_tgt_imu = t_imu[mask_imu].astype(np.float64)

        # prepare overlap views for mag/baro before nearest selection
        mb_overlap   = _mask_dict_by_len(matched_mag_baro_data, mask_mb,   len(t_mb))

        # nearest-neighbor align both streams to IMU times
        mb_cut   = _align_by_nearest(t_mb[mask_mb],     t_tgt_imu, mb_overlap)

        return imu_cut, matched_cut, mb_cut
        
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
    
    def _transform_to_cpf_world(self, imu_data, matched_data, matched_mag_baro_data):
        """Transform all data with corrected transformation chain"""
        n_samples = len(imu_data['timestamps'])
        
        # Prepare output arrays
        orientations_cpfworld = []
        positions_cpfworld = []
        velocities_cpf = []
        
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

            # Build world->cpfworld homogeneous transform once
            R_cw = self.R_world_to_cpfworld.T                   # 3x3
            t_cw = - R_cw @ self.reference_position             # 3,
            T_cw = np.eye(4)
            T_cw[:3, :3] = R_cw
            T_cw[:3, 3]  = t_cw
        
        return {
            'timestamps': imu_data['timestamps'],

            # Right IMU in CPF frame
            'accelerometer_cpf': imu_data['accelerometer_cpf'],
            'gyroscope_cpf': imu_data['gyroscope_cpf'],

            # Left IMU in CPF frame (for ablation/fusion)
            'accel_left_cpf': imu_data.get('accel_left_cpf', None),
            'gyro_left_cpf': imu_data.get('gyro_left_cpf', None),

            # Left IMU in its own body frame (for ablation studies)
            'accel_left_body': imu_data.get('accel_left_body', None),
            'gyro_left_body': imu_data.get('gyro_left_body', None),

            # Ground truth
            'orientation_cpfworld': np.array(orientations_cpfworld, dtype=np.float32),
            'position_cpfworld':    np.array(positions_cpfworld,    dtype=np.float32),
            'velocity_cpf':         np.array(velocities_cpf,        dtype=np.float32),

            # Mag/Baro
            'mag':       matched_mag_baro_data.get('mag_cali', None),
            'pressure':  matched_mag_baro_data.get('pressure_cali', None),
            'altitude':  matched_mag_baro_data.get('altitude_cali', None),
            'temperature': matched_mag_baro_data.get('temperature', None),
        }
    
    def _resample_data_single_rate(self, data, sample_hz):
        """Resample all data to single rate (simplified, like Aria)

        Args:
            data: Input data dict (all arrays aligned to data['timestamps'])
            sample_hz: Target Hz for all data (default 200Hz)
        """
        import numpy as np
        from scipy.spatial.transform import Rotation, Slerp

        t_src = np.asarray(data['timestamps'], dtype=np.float64)
        t_start, t_end = t_src[0], t_src[-1]

        # Create single time grid
        time_resampled = np.arange(t_start, t_end, 1.0 / sample_hz)
        n_samples = len(time_resampled)

        print(f"   Resampling from {len(t_src)} samples to {n_samples} samples @ {sample_hz} Hz")

        resampled = {}

        # ---------- helpers ----------
        def _interp_vec3(tdst, ts, X):
            """Interpolate (N,3) to (len(tdst),3)."""
            out = np.empty((len(tdst), 3), dtype=np.float32)
            for k in range(3):
                out[:, k] = np.interp(tdst, ts, X[:, k].astype(np.float64)).astype(np.float32)
            return out

        def _interp_scalar(tdst, ts, x):
            """Interpolate (N,) to (len(tdst),)."""
            return np.interp(tdst, ts, x.astype(np.float64)).astype(np.float32)

        # ---------- Right IMU ----------
        for key in ['accelerometer_cpf', 'gyroscope_cpf']:
            if key in data and data[key] is not None:
                resampled[key] = _interp_vec3(time_resampled, t_src, np.asarray(data[key], dtype=np.float32))

        # ---------- Left IMU (CPF frame) ----------
        for key in ['accel_left_cpf', 'gyro_left_cpf']:
            if key in data and data[key] is not None:
                resampled[key] = _interp_vec3(time_resampled, t_src, np.asarray(data[key], dtype=np.float32))

        # ---------- Left IMU (body frame) ----------
        for key in ['accel_left_body', 'gyro_left_body']:
            if key in data and data[key] is not None:
                resampled[key] = _interp_vec3(time_resampled, t_src, np.asarray(data[key], dtype=np.float32))

        # ---------- Pose (position/velocity) ----------
        for key in ['position_cpfworld', 'velocity_cpf']:
            if key in data and data[key] is not None:
                resampled[key] = _interp_vec3(time_resampled, t_src, np.asarray(data[key], dtype=np.float32))

        # ---------- Orientation (quaternions) ----------
        if 'orientation_cpfworld' in data and data['orientation_cpfworld'] is not None:
            q_src = np.asarray(data['orientation_cpfworld'], dtype=np.float64)
            q_out = np.zeros((n_samples, 4), dtype=np.float64)
            for i in range(4):
                q_out[:, i] = np.interp(time_resampled, t_src, q_src[:, i])
            q_out /= (np.linalg.norm(q_out, axis=1, keepdims=True) + 1e-12)
            resampled['orientation_cpfworld'] = q_out.astype(np.float32)

        # ---------- Mag / Baro ----------
        if 'mag' in data and data['mag'] is not None:
            resampled['mag'] = _interp_vec3(time_resampled, t_src, np.asarray(data['mag'], dtype=np.float32))

        for k_src, k_out in [
            ('pressure',   'pressure'),
            ('altitude',   'altitude'),
            ('temperature','temperature'),
        ]:
            if k_src in data and data[k_src] is not None:
                resampled[k_out] = _interp_scalar(time_resampled, t_src, np.asarray(data[k_src], dtype=np.float32))

        # ---------- time & metadata ----------
        resampled['timestamps'] = time_resampled.astype(np.float32)
        resampled['sampling_rate_hz'] = float(sample_hz)
        resampled['num_samples'] = int(n_samples)

        return resampled

    def _validate_integration(self, data):
        """Validate velocity integration with corrected transformation"""
        # Use resampled timestamps for validation
        timestamps = data['timestamps']
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
    sequence_path, output_dir, skip_seconds, sample_hz = args


    output_path = output_dir / f"{sequence_path.name}_cpfbody_{sample_hz}hz.pkl"
    if output_path.exists():
        print("File already exists:", output_path)
        return None

    try:
        transformer = CPFWorldTransformer(sequence_path, skip_seconds)
        data = transformer.process_full_sequence(sample_hz)
        
        if data is None:
            return None
        
        # Prepare output
        output_data = {
            # Metadata
            'sequence_name': sequence_path.name,
            'duration_seconds': float(data['timestamps'][-1] - data['timestamps'][0]),
            'num_samples': int(data['num_samples']),
            'sampling_rate_hz': float(data['sampling_rate_hz']),
            
            # Frame info
            'coordinate_frames': {
                'imu_right': 'CPF (body frame)',
                'imu_left_cpf': 'CPF (body frame) - same as right',
                'imu_left_body': 'Left IMU own body frame (after calibration, before CPF transform)',
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
            
            # Time-series data as torch tensors (single rate, simplified like Aria)
            'imu_data': {
                'time': torch.tensor(data['timestamps']),

                # Right IMU in CPF frame (default)
                'accel': torch.tensor(data['accelerometer_cpf']),
                'gyro':  torch.tensor(data['gyroscope_cpf']),

                # Left IMU in CPF frame (for ablation/fusion in dataloader)
                'accel_left_cpf':  torch.tensor(data['accel_left_cpf'])  if 'accel_left_cpf'  in data and data['accel_left_cpf']  is not None else None,
                'gyro_left_cpf':   torch.tensor(data['gyro_left_cpf'])   if 'gyro_left_cpf'   in data and data['gyro_left_cpf']   is not None else None,

                # Left IMU in its own body frame (for ablation studies)
                'accel_left_body': torch.tensor(data['accel_left_body']) if 'accel_left_body' in data and data['accel_left_body'] is not None else None,
                'gyro_left_body':  torch.tensor(data['gyro_left_body'])  if 'gyro_left_body'  in data and data['gyro_left_body']  is not None else None,

                # Mag/Baro
                'mag':        torch.tensor(data['mag'])        if 'mag'        in data and data['mag']        is not None else None,
                'pressure':   torch.tensor(data['pressure'])   if 'pressure'   in data and data['pressure']   is not None else None,
                'altitude':   torch.tensor(data['altitude'])   if 'altitude'   in data and data['altitude']   is not None else None,
                'temperature':torch.tensor(data['temperature'])if 'temperature'in data and data['temperature']is not None else None,
            },
            'gt_data': {
                'time': torch.tensor(data['timestamps']),
                'orientation': torch.tensor(data['orientation_cpfworld']),
                'position': torch.tensor(data['position_cpfworld']),
                'velocity': torch.tensor(data['velocity_cpf']),
            },
            'validation': data['validation']
        }
        
        # Save
        output_path = output_dir / f"{sequence_path.name}_cpfbody_{sample_hz}hz.pkl"
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
    parser.add_argument("--sample-hz", type=int, default=200, help="Target sampling rate in Hz (default 200)")
    parser.add_argument("--parallel", type=int, default=16, help="Number of parallel workers")
    parser.add_argument("--specificlist", help="Process a list of specific sequence by name")
    args = parser.parse_args()

    input_path = Path(args.path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print("="*80)
    print("🚀 CORRECTED CPF TRANSFORMATION PIPELINE (SIMPLIFIED)")
    print("="*80)
    print(f"📂 Input: {input_path}")
    print(f"📂 Output: {output_dir}")
    print(f"⏰ Sampling Rate: {args.sample_hz} Hz")
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
        result = process_sequence_wrapper((sequences[0], output_dir, args.skip_seconds, args.sample_hz))
        success_count = 1 if result else 0
        failed_sequences = [] if result else [sequences[0].name]
    else:
        # Parallel processing
        print(f"🔧 Processing with {args.parallel} workers...")
        process_args = [(seq, output_dir, args.skip_seconds, args.sample_hz) for seq in sequences]
        
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


# Example usage:
# python preprocess_nymeria_mag_baro_2imu_both.py $DATA_ROOT/nymeria_head_full -o $DATA_ROOT/processed_nymeria_2imu_both_testing_Apr_3 --sample-hz 200 --max 1