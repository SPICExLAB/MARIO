import os
import pickle
import numpy as np
import torch
import pypose as pp
from .dataset import Sequence
import matplotlib.pyplot as plt
import torch.nn.functional as F

from datasets.dataset_utils import truncate_and_free, smoothing


class ariaCPF(Sequence):
    def __init__(
        self,
        data_root: str,
        data_name: str,
        coordinate: str = None,
        mode: str = None,
        gravity: float = 9.81,
        remove_g: bool = False,
        maximum_length: int = 5000000000,
        load_baro: bool = True,
        load_mag: bool = True,
        load_left_imu: bool = True,
        load_left_imu_cpf: bool = False,
        load_left_imu_body: bool = False,
        rot_from_gyro: bool = False,
        imu_in_body_frame: bool = False,
        **kwargs
    ):
        super().__init__()
        # Load pickle file - try .pkl first, then .pickle
        import time
        t1 = time.perf_counter()
        pkl_file = os.path.join(data_root, data_name + '.pkl')
        if not os.path.exists(pkl_file):
            pkl_file = os.path.join(data_root, data_name + '.pickle')
        with open(pkl_file, 'rb') as f:
            raw = pickle.load(f)
        elapsed = time.perf_counter() - t1
        print(f"loading took {elapsed:.3f} s")
        
        # Initialize data dictionary
        self.data = {}
        
        # Process IMU data
        if 'imu_data' in raw:
 
            # Single-rate: use common timestamps
            if 'time' in raw['imu_data']:
                self.data['time'] = raw['imu_data']['time']
            
            # Handle accelerometer data - already in CPF frame from preprocessing
            if 'accel' in raw['imu_data']:
                self.data['acc'] = raw['imu_data']['accel'].float()
            
            # Handle gyroscope data - already in CPF frame from preprocessing
            if 'gyro' in raw['imu_data']:
                self.data['gyro'] = raw['imu_data']['gyro'].float() # world frame gyro
        
        # Process ground truth data
        if 'gt_data' in raw:
            
            # Single-rate: use common timestamps
            if 'time' not in self.data and 'time' in raw['gt_data']:
                self.data['time'] = raw['gt_data']['time']
                
            # Handle orientation data - now stored as CPF→World quaternions
            if 'orientation' in raw['gt_data']:
                orientation = raw['gt_data']['orientation'].float()
                if isinstance(orientation, torch.Tensor) and not isinstance(orientation, pp.LieTensor):
                    # Check if it's a quaternion [qx, qy, qz, qw] (PyPose order)
                    if orientation.shape[-1] == 4:
                        # Convert to PyPose SO3 - represents CPF→World rotation
                        self.data['gt_orientation'] = pp.SO3(orientation)
                        print(f"   ✓ Loaded CPF→World orientations: {orientation.shape}")
                    else:
                        print(f"Warning: Unexpected orientation shape: {orientation.shape}")
                        self.data['gt_orientation'] = orientation
                else:
                    self.data['gt_orientation'] = orientation
            
            # Handle position data
            if 'position' in raw['gt_data']:
                self.data['gt_translation'] = raw['gt_data']['position'].float()
                
            # Handle velocity data - already in CPF frame from preprocessing
            if 'velocity' in raw['gt_data']:
                self.data['velocity'] = raw['gt_data']['velocity'].float()

        # Single-rate dt calculation
        if 'dt' in raw.get('imu_data', {}):
            self.data['dt'] = raw['imu_data']['dt']
        else:
            time = self.data['time']
            dt = torch.zeros_like(time)
            if len(time) > 1:
                dt[1:] = time[1:] - time[:-1]
                dt[0] = dt[1]  # Set first dt to second dt
            self.data['dt'] = dt

        if 'altitude' in raw['imu_data'] and load_baro:
            self.data['altitude'] = raw['imu_data']['altitude']
            if len(self.data['altitude'].shape) == 2:
                self.data['altitude'] = self.data['altitude'][:,0]
            self.data['altitude'] = torch.Tensor(smoothing(self.data['altitude'][:], kernel_size=1001))
            
            altitude = self.data['altitude']           
            alt_vel = torch.zeros_like(altitude)
            alt_vel[1:] = (altitude[1:] - altitude[:-1]) / self.data['dt'][1:]
            alt_vel[0] = alt_vel[1]  # match first frame
            alt_vel = torch.Tensor(smoothing(alt_vel, kernel_size=101))
            self.data['baro'] = alt_vel.unsqueeze(-1)
            
            baro = self.data['baro'].flatten()

            bad = ~torch.isfinite(baro)
            if bad.any().item(): 
                self.data['baro'] = torch.zeros_like(self.data['baro'])

            self.data['temperature'] = raw['imu_data']['temperature'] # [N]
                
        if 'mag' in raw['imu_data'] and load_mag:
            mag_acc = np.array([0, -gravity, 0])
            mag = raw['imu_data']['mag'].detach().cpu().numpy()

            if coordinate == "glob_coord":
                # rotate world gravity into BODY/CPF for each frame (N,3)
                mag_acc = self.data['gt_orientation'].Inv() @ torch.from_numpy(mag_acc).float()
            else:
                # expand to (N,3) to match mag for per-row normalization
                mag_acc = np.broadcast_to(mag_acc, mag.shape)

            # Normalize vectors
            mag_acc_norm = mag_acc / (np.linalg.norm(mag_acc, axis=1, keepdims=True) + 1e-8)
            mag_norm     = mag     / (np.linalg.norm(mag,     axis=1, keepdims=True) + 1e-8)

            # Compute horizontal components using tilt compensation
            east = np.cross(mag_norm, mag_acc_norm)
            east /= (np.linalg.norm(east, axis=1, keepdims=True) + 1e-8)
            north = np.cross(mag_acc_norm, east)

            # Yaw (about +Y) in the X–Z plane for z-forward, y-up, x-left
            yaw = np.arctan2(east[:, 2], north[:, 2])   # radians
            self.data['mag'] = torch.from_numpy(yaw).unsqueeze(-1).float()

        # if load_left_imu and 'accel_left' in raw['imu_data']:
        #     self.data['acc_left'] = raw['imu_data']['accel_left'].float()
        #     self.data['gyro_left'] = raw['imu_data']['gyro_left'].float()
        
        # Backward compatibility: if load_left_imu is True but neither cpf nor body specified, use cpf
        if load_left_imu and not load_left_imu_cpf and not load_left_imu_body:
            load_left_imu_cpf = True

        # Check mutual exclusivity
        if load_left_imu_cpf and load_left_imu_body:
            raise ValueError("Cannot load both CPF and body frames simultaneously for left IMU")

        # Load CPF frame
        if load_left_imu_cpf:
            if 'accel_left_cpf' in raw['imu_data']:
                self.data['acc_left'] = raw['imu_data']['accel_left_cpf']
                self.data['gyro_left'] = raw['imu_data']['gyro_left_cpf']
            elif 'accel_left' in raw['imu_data']:  # Backward compatibility
                self.data['acc_left'] = raw['imu_data']['accel_left']
                self.data['gyro_left'] = raw['imu_data']['gyro_left'] 
        # Load body frame
        elif load_left_imu_body:
            if 'accel_left_body' not in raw['imu_data']:
                raise ValueError("Body frame not found in pickle file. Use preprocessing with '_both' suffix.")
            self.data['acc_left'] = raw['imu_data']['accel_left_body']
            self.data['gyro_left'] = raw['imu_data']['gyro_left_body']
                        
        # Create mask (all valid by default)
        mask_size = len(self.data.get('time_pose', self.data.get('time', [])))
        self.data['mask'] = torch.ones(mask_size, dtype=torch.bool)
                
        # Apply coordinate transform if specified
        if coordinate:
            self.update_coordinate(coordinate, mode)
        
        assert coordinate in [None, "glob_coord", "body_coord"], "Coordinate must be None, 'glob_coord' or 'body_coord'"
        self.coordinate = coordinate
        
        # Store metadata
        self.gravity = torch.tensor([0,-gravity,0],dtype=self.data['acc'].dtype)
        self.gravity = self.gravity.expand(self.data['acc'].shape[0], -1)
        
        if remove_g:
            self.data['acc'] = self.data['acc'] + self.data["gt_orientation"].Inv() @ self.gravity
            
            if "acc_left" in self.data and load_left_imu_cpf:
                self.data['acc_left'] = self.data['acc_left'] + self.data["gt_orientation"].Inv() @ self.gravity
        
        ##################### Rotation from gyro integration #############################
        if rot_from_gyro:
            # Right IMU: integrate gyro to get orientation (CPF frame)
            R0   = self.data['gt_orientation'][0]                  # [] SO3 (LieTensor)
            gyro = self.data['gyro'][:-1]                          # [N-1, 3]
            dt   = self.data['dt'].reshape(-1) if self.data['dt'].ndim > 1 else self.data['dt']  # [N]
            dt   = dt[:-1]                                         # [N-1] to match gyro

            # keep everything on the same device/dtype
            gyro = gyro.to(R0.device)
            dt   = dt.to(R0.device)

            theta = gyro * dt.unsqueeze(-1)                        # [N-1, 3] radians
            delta = pp.so3(theta).Exp()                            # [N-1] SO3 increments
            Rrel  = pp.cumprod(delta, dim=0, left=False)           # [N-1] SO3 (prefix products)

            # include the initial pose as the first element; total length N
            Rseq = torch.cat([R0.unsqueeze(0), (R0 * Rrel)], dim=0)  # [N] SO3

            self.data['gt_orientation'] = Rseq

        # Left IMU: integrate gyro_left to get orientation_left (CPF or body frame)
        if (load_left_imu_cpf or load_left_imu_body) and 'gyro_left' in self.data:
            # Use identity as initial orientation for left IMU (in its own reference frame)
            # Or use same initial orientation as right IMU if both in CPF
            if load_left_imu_cpf:
                # Left IMU in CPF frame - use same initial orientation as right
                R0_left = self.data['gt_orientation'][0]
            else:
                # Left IMU in body frame - start from identity (no initial rotation)
                # R0_left = pp.identity_SO3(1).to(self.data['gyro_left'].device)
                R0_left = self.data['gt_orientation'][0]

            gyro_left = self.data['gyro_left'][:-1]  # [N-1, 3]
            dt = self.data['dt'].reshape(-1) if self.data['dt'].ndim > 1 else self.data['dt']
            dt = dt[:-1]

            gyro_left = gyro_left.to(R0_left.device)
            dt = dt.to(R0_left.device)

            theta_left = gyro_left * dt.unsqueeze(-1)
            delta_left = pp.so3(theta_left).Exp()
            Rrel_left = pp.cumprod(delta_left, dim=0, left=False)

            Rseq_left = torch.cat([R0_left.unsqueeze(0), (R0_left * Rrel_left)], dim=0)
            self.data['orientation_left'] = Rseq_left
            print(f"   ✓ Integrated left IMU gyro to orientation_left: {Rseq_left.shape}")

        ###################################################################################

        # Rotate IMU to world frame only if coordinate is glob_coord AND imu_in_body_frame is False
        # When imu_in_body_frame=True, keep IMU in body frame to feed directly to body model
        if self.coordinate == "glob_coord" and not imu_in_body_frame:
            self.data['acc'] = self.data["gt_orientation"] @ self.data['acc']

            # Transform left IMU to world frame if available
            if 'acc_left' in self.data and load_left_imu_cpf:
                self.data['acc_left'] = self.data["gt_orientation"] @ self.data['acc_left']
        

        keys = [
            "velocity","gt_translation","gt_orientation","time","mask","dt","acc","gyro",
            'baro', "temperature", "acc_left", "gyro_left", "mag", "orientation_left"
        ]
        truncate_and_free(self.data, keys, maximum_length)

        # Print summary
        print(f"\n=== Loaded Nymeria dataset: {data_name} ===")
        
        print(f"Mode: SINGLE-RATE")
        single_rate = raw.get('sampling_rate_hz', 'unknown')
        print(f"Total frames: {self.get_length()} @ {single_rate}Hz")
        print(f"Coordinate system: {coordinate if coordinate else 'original (CPF frame)'}")
        print(f"Gravity removal: {remove_g}")
        print(f"\nData summary:")
        for key, value in self.data.items():
            if hasattr(value, 'shape'):
                print(f"  - {key}: shape={value.shape}, dtype={value.dtype}")
            else:
                print(f"  - {key}: {type(value)}")
        
    def get_length(self) -> int:
        """Number of samples equals number of time stamps"""
        
        return self.data['time'].shape[0]

    def update_coordinate(self, coordinate: str, mode: str):
        """
        Updates the data based on the required coordinate system.
        
        NOTE: Raw data from preprocessing is now:
        - IMU (acc, gyro) in CPF frame
        - Velocity in CPF frame  
        - Position in world frame
        - Orientation quaternions represent CPF→World rotations
        - Gravity in world frame
        
        Args:
            coordinate: 'glob_coord' or 'body_coord'
            mode: Dataset mode ('train', 'inference', etc.)
        """
        if coordinate is None:
            print("No coordinate system provided. Using original CPF frame.")
            return
            
        print(f"\nUpdating coordinate system to: {coordinate}")
        
        try:
            if coordinate == "glob_coord":
                # Transform CPF velocities to world frame using orientations
                print("Transforming CPF velocities to world frame using orientations...")
                
                # Get CPF velocities (body frame)
                cpf_velocities = self.data["velocity"].clone()  # Clone to preserve original
                print(f"CPF velocity stats before transformation:")
                cpf_vel_magnitudes = torch.norm(cpf_velocities, dim=1)
                print(f"  Min: {cpf_vel_magnitudes.min():.3f} m/s")
                print(f"  Max: {cpf_vel_magnitudes.max():.3f} m/s")
                print(f"  Mean: {cpf_vel_magnitudes.mean():.3f} m/s")
                
                # Get CPF→World orientations and convert to rotation matrices
                orientations = self.data["gt_orientation"]
                R_cpf_to_world = orientations.matrix().float()  # Shape: [N, 3, 3]
                
                # Vectorized transformation: v_world = R @ v_cpf
                # cpf_velocities: [N, 3] -> [N, 3, 1]
                # R @ v: [N, 3, 3] @ [N, 3, 1] -> [N, 3, 1] -> [N, 3]
                world_velocities = torch.bmm(R_cpf_to_world, cpf_velocities.unsqueeze(-1)).squeeze(-1)
                
                # Replace velocity with world frame velocity
                self.data["velocity"] = world_velocities
            
            elif coordinate == "body_coord":
                # For body_coord: data already in CPF/body frame
                print("Data already in CPF/body frame")
                # Network was trained on CPF velocities, so evaluation should compare CPF vs CPF
                print("✓ Velocity remains in CPF frame for consistent evaluation")
            else:
                raise ValueError(f"Unsupported coordinate system: {coordinate}")
                
        except Exception as e:
            print(f"Error during coordinate transformation: {e}")
            raise e

    
if __name__ == "__main__":
    data_root=os.path.join(os.environ.get("DATA_ROOT", "."), "aria_processed_cpf_both/test")
    data_name="loc1_script5_seq6_rec1_cpfbody_200hz"
    
    coordinate="glob_coord"

    dataset = ariaCPF(
        data_root=data_root,
        data_name=data_name,
        coordinate=coordinate,
        remove_g=True,
        maximum_length=80000000,
        load_baro=True,
        load_mag=True,
        load_left_imu=True
    )

   
# python -m datasets.ariaCPFdataset