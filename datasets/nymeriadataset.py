import os
import pickle
import numpy as np
import torch
import pypose as pp
from .dataset import Sequence
import matplotlib.pyplot as plt
import torch.nn.functional as F

from datasets.dataset_utils import truncate_and_free, smoothing


# ----------------------------------------------------------------------------
# Body-pose helpers (PoseNet training labels). Ported from the IMUSLAM_clean
# branch; only used when the dataset config sets `use_body: True`.
# ----------------------------------------------------------------------------
def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """Rotation matrices (*, 3, 3) -> 6D representation (*, 6) [Zhou et al. 2019]."""
    batch_dim = matrix.size()[:-2]
    return matrix[..., :2, :].clone().reshape(batch_dim + (6,))


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """6D representation (*, 6) -> rotation matrices (*, 3, 3) via Gram-Schmidt."""
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


# SMPL joint index -> Xsens segment index
XSENS_TO_SMPL = [
    0,   # 0  pelvis           -> Pelvis
    19,  # 1  left_hip         -> LUpperLeg
    15,  # 2  right_hip        -> RUpperLeg
    1,   # 3  spine1           -> L5
    20,  # 4  left_knee        -> LLowerLeg
    16,  # 5  right_knee       -> RLowerLeg
    2,   # 6  spine2           -> L3
    21,  # 7  left_ankle       -> LFoot
    17,  # 8  right_ankle      -> RFoot
    3,   # 9  spine3           -> T12
    22,  # 10 left_foot        -> LToe
    18,  # 11 right_foot       -> RToe
    5,   # 12 neck             -> Neck
    11,  # 13 left_collar      -> LShoulder
    7,   # 14 right_collar     -> RShoulder
    6,   # 15 head             -> Head
    12,  # 16 left_shoulder    -> LUpperArm
    8,   # 17 right_shoulder   -> RUpperArm
    13,  # 18 left_elbow       -> LForearm
    9,   # 19 right_elbow      -> RForearm
    14,  # 20 left_wrist       -> LHand
    10,  # 21 right_wrist      -> RHand
    14,  # 22 left_hand        -> LHand
    10,  # 23 right_hand       -> RHand
]


def xsens_to_SMPL(xsens_rotmat):
    """Xsens segment rotations (N, 23, 3, 3) -> SMPL global joint rotations (N, 24, 3, 3)."""
    perm = [1, 2, 0]
    xsens_rotmat = xsens_rotmat[:, :, perm, :]
    xsens_rotmat = xsens_rotmat[:, :, :, perm]
    xsens_rotmat = torch.as_tensor(xsens_rotmat, dtype=torch.float32).clone()

    Rx90 = torch.tensor([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]], dtype=xsens_rotmat.dtype)
    Rz90 = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]], dtype=xsens_rotmat.dtype)
    xsens_rotmat = Rz90 @ (Rx90 @ xsens_rotmat)

    smpl_rotmat = torch.eye(3).repeat(xsens_rotmat.shape[0], 24, 1, 1)
    for smpl_idx, xsens_idx in enumerate(XSENS_TO_SMPL):
        smpl_rotmat[:, smpl_idx] = xsens_rotmat[:, xsens_idx]
    return smpl_rotmat


def normalize_rows(x, eps=1e-9):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (n + eps)

class nymeria(Sequence):
    def __init__(
        self,
        data_root: str,
        data_name: str,
        coordinate: str = None,
        mode: str = None,
        gravity: float = 9.81,
        remove_g: bool = False,
        maximum_length: int = 150000000,
        root_vel_as_vel: bool = False,
        load_baro: bool = True,
        load_mag: bool = True,
        load_left_imu: bool = True,
        load_left_imu_cpf: bool = True,  # NEW: Load left IMU in CPF frame
        load_left_imu_body: bool = False,  # NEW: Load left IMU in body frame
        imu_in_body_frame: bool = False,  # Keep IMU in body frame even with glob_coord
        rot_from_gyro: bool = False,
        resample_hz: float = 200,
        use_body: bool = False,  # derive SMPL joint labels from Xsens pose (PoseNet training)
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
                self.data['acc'] = raw['imu_data']['accel']
            
            # Handle gyroscope data - already in CPF frame from preprocessing
            if 'gyro' in raw['imu_data']:
                self.data['gyro'] = raw['imu_data']['gyro']
        
        # Process ground truth data
        if 'gt_data' in raw:
            
            # Single-rate: use common timestamps
            if 'time' not in self.data and 'time' in raw['gt_data']:
                self.data['time'] = raw['gt_data']['time']
            
            # Handle orientation data - now stored as CPF→World quaternions
            if 'orientation' in raw['gt_data']:
                orientation = raw['gt_data']['orientation']
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
                
                # self.data['gt_orientation_yaw_only'] = self._heading_only_from_so3(self.data['gt_orientation'])
            
            # Handle position data
            if 'position' in raw['gt_data']:
                self.data['gt_translation'] = raw['gt_data']['position']
                
            # Handle velocity data - already in CPF frame from preprocessing
            if 'velocity' in raw['gt_data']:
                self.data['velocity'] = raw['gt_data']['velocity']
                
            ############################ Body pose labels (PoseNet training) ############################
            if use_body and 'xsens_pose' in raw['gt_data']:
                import core.articulate as art
                from core.paths import Paths
                if not Paths.SMPL_FILE.exists():
                    raise FileNotFoundError(
                        f"SMPL model not found at {Paths.SMPL_FILE}. Download basicmodel_m.pkl from "
                        "https://smpl.is.tue.mpg.de and place it there or set SMPL_MODEL_PATH.")
                xsens_pose = raw['gt_data']['xsens_pose']                       # [N, 23, 4, 4] segments in CPF-world
                if not isinstance(xsens_pose, torch.Tensor):
                    xsens_pose = torch.tensor(np.asarray(xsens_pose), dtype=torch.float32)
                self.data['body_translation'] = xsens_pose[:, 0, :3, 3]        # pelvis position [N, 3]
                body_smpl = xsens_to_SMPL(xsens_pose[:, :, :3, :3])            # [N, 24, 3, 3] global rotations
                fk_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                body_model = art.model.ParametricModel(Paths.SMPL_FILE, device=fk_device)
                local_poses = body_model.inverse_kinematics_R(body_smpl).view(body_smpl.shape[0], 24, 3, 3)
                # Root-relative joint positions: forward kinematics without the root translation,
                # so PoseNet predicts body configuration rather than world position.
                _, joint_global = body_model.forward_kinematics(local_poses.to(fk_device), calc_mesh=False)
                self.data['joint_pos'] = joint_global.flatten(start_dim=1).cpu()                        # [N, 24*3]
                self.data['joint_rot'] = matrix_to_rotation_6d(local_poses).flatten(start_dim=1).cpu()  # [N, 24*6]
                del body_model, joint_global
                print(f"   ✓ Derived SMPL joint labels: joint_pos {tuple(self.data['joint_pos'].shape)}, "
                      f"joint_rot {tuple(self.data['joint_rot'].shape)}")

        self.device = self.data['velocity'].device
        
        if root_vel_as_vel:
            self.data['velocity'] = self.data['root_velocity']
        
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
            if len(self.data['altitude'].size()) == 2:
                self.data['altitude'] = self.data['altitude'][:, 0]
            self.data['altitude'] = torch.Tensor(smoothing(self.data['altitude'][:], kernel_size=1001))
            
            altitude = self.data['altitude']           
            alt_vel = torch.zeros_like(altitude)
            alt_vel[1:] = (altitude[1:] - altitude[:-1]) / self.data['dt'][1:]
            alt_vel[0] = alt_vel[1]  # match first frame
            alt_vel = torch.Tensor(smoothing(alt_vel, kernel_size=101))
            self.data['baro'] = alt_vel.unsqueeze(-1)
            
            self.data['temperature'] = raw['imu_data']['temperature'] # [N]
        
        Rseq = None
        # ##################### Rotation from gyro integration #############################
        # Right IMU: integrate gyro to get orientation (CPF frame)
        R0   = self.data['gt_orientation'][0]                  # [] SO3 (LieTensor)
        gyro = self.data['gyro'][:-1]                          # [N-1, 3] - Right IMU
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

        self.data['orientation_from_gyro'] = Rseq

        print(f"   ✓ Replaced GT orientations with GYRO-INTEGRATED rotations: {Rseq.shape}")
            
        if 'mag' in raw['imu_data'] and load_mag:
  
            mag_sensor = raw['imu_data']['mag']
            
            R_dev2cpf = raw['R_device_to_CPF_vectors']  # torch, shape [3,3]
            # (Pdb) p R_dev2cpf
            # array([[-3.1860135e-02,  7.9335338e-01,  6.0792708e-01],
            #        [-9.9862951e-01,  1.7481547e-09, -5.2336007e-02],
            #        [-4.1520949e-02, -6.0876137e-01,  7.9226613e-01]], dtype=torch.float32)

            R_mag2dev = torch.tensor([
                [ -0.9999086,   0.0311673,  -0.0028411 ],
                [  0.0279477,   0.9601649,  -0.2779936 ],
                [ -0.0100836,  -0.2778216,  -0.9605636 ]
            ], dtype=torch.float32) # mag to device frame
            mag_dev = mag_sensor @ R_mag2dev.T                  # mag -> device
            mag_cpf = mag_dev @ R_dev2cpf.T                     # device -> CPF

            self.data['mag_vec'] = mag_cpf 
            
            g_world = torch.tensor([0.0, -gravity, 0.0], dtype=mag_cpf.dtype, device=mag_cpf.device)
            # g_cpf = (self.data['orientation_from_gyro'].Inv() @ g_world)   # torch [N,3]
            g_cpf = (self.data['gt_orientation'].Inv() @ g_world)   # torch [N,3]
                        
            # g_cpf = self.data['acc']

            # to numpy for cross products (or keep in torch; either is fine)
            mag_np = mag_cpf.detach().cpu().numpy()     # [N,3]
            g_np   = g_cpf.detach().cpu().numpy()       # [N,3]

            mag_norm = mag_np / (np.linalg.norm(mag_np, axis=1, keepdims=True) + 1e-8)
            g_norm   = g_np   / (np.linalg.norm(g_np,   axis=1, keepdims=True) + 1e-8)

            east  = np.cross(mag_norm, g_norm)
            east /= (np.linalg.norm(east, axis=1, keepdims=True) + 1e-8)
            north = np.cross(g_norm, east)  # already unit-length-ish

            yaw_mag = np.arctan2(north[:, 0], north[:, 2]).astype(np.float32)  # radians
            self.data['mag'] = torch.from_numpy(-yaw_mag).unsqueeze(-1)  # [N,1]
                
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

            # Integrate left gyro (same R0 and dt as right gyro)
            if 'gyro_left' in self.data:
                gyro_left = self.data['gyro_left'][:-1].to(R0.device)
                theta_left = gyro_left * dt.unsqueeze(-1)
                delta_left = pp.so3(theta_left).Exp()
                Rrel_left  = pp.cumprod(delta_left, dim=0, left=False)
                Rseq_left  = torch.cat([R0.unsqueeze(0), (R0 * Rrel_left)], dim=0)
                self.data['orientation_from_gyro_left'] = Rseq_left
                print(f"   ✓ Computed left gyro integrated rotations: {Rseq_left.shape}")

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

            # Also remove gravity from left IMU if available
            if load_left_imu_cpf and 'acc_left' in self.data:
                self.data['acc_left'] = self.data['acc_left'] + self.data["gt_orientation"].Inv() @ self.gravity

        # IMU transformation: controlled separately from velocity frame
        if self.coordinate == "glob_coord" and not imu_in_body_frame:
            # Rotate IMU to world frame (old behavior)
            self.data['acc'] = self.data["gt_orientation"] @ self.data['acc']

            # Transform left IMU to world frame if available
            if 'acc_left' in self.data and load_left_imu_cpf:
                self.data['acc_left'] = self.data["gt_orientation"] @ self.data['acc_left']

        if Rseq is not None and rot_from_gyro:
            self.data['gt_orientation'] = Rseq
        
        keys = [
            "velocity","gt_translation","gt_orientation","time","mask","dt","acc","gyro",
         'baro', "temperature", "acc_left", "gyro_left", "mag", "mag_vec", "orientation_left",
         "joint_pos", "joint_rot", "body_translation"

        ]
        truncate_and_free(self.data, keys, maximum_length)

        if resample_hz is not None:
            # source time
            t_src = self.data["time"].reshape(-1).to(self.device)

            # build uniform 50Hz grid in seconds
            T0 = t_src[0]
            t0 = 0.0
            t_end = float((t_src[-1] - T0).item())
            dt_tgt = 1.0 / float(resample_hz)

            # M samples from 0..t_end inclusive
            M = int(torch.floor(torch.tensor(t_end / dt_tgt)).item()) + 1
            t_rel_tgt = torch.arange(M, device=self.device, dtype=t_src.dtype) * dt_tgt  # [M]
            t_tgt = T0 + t_rel_tgt  # [M] in same timebase as t_src

            def has(k): return (k in self.data) and (self.data[k] is not None)

            # --- orientation: SO3 slerp ---
            if has("gt_orientation") and isinstance(self.data["gt_orientation"], pp.LieTensor):
                self.data["gt_orientation"] = self._resample_so3_slerp(self.data["gt_orientation"], t_src, t_tgt)
            
            if has("orientation_from_gyro") and isinstance(self.data["orientation_from_gyro"], pp.LieTensor):
                self.data["orientation_from_gyro"] = self._resample_so3_slerp(self.data["orientation_from_gyro"], t_src, t_tgt)

            if has("orientation_from_gyro_left") and isinstance(self.data["orientation_from_gyro_left"], pp.LieTensor):
                self.data["orientation_from_gyro_left"] = self._resample_so3_slerp(self.data["orientation_from_gyro_left"], t_src, t_tgt)

            # --- vector series (linear) ---
            vec_keys = ["acc", "gyro", "velocity", "gt_translation", "acc_left", "gyro_left", "mag_vec",
                        "joint_pos", "joint_rot", "body_translation"]
            for k in vec_keys:
                if has(k):
                    x = self.data[k]
                    if isinstance(x, torch.Tensor):
                        self.data[k] = self._interp1d_torch(t_src, x, t_tgt)

            # --- scalar series (linear) ---
            scalar_keys = ["altitude", "baro", "temperature", "mag"]
            for k in scalar_keys:
                if has(k):
                    x = self.data[k]
                    if isinstance(x, torch.Tensor):
                        x = x.reshape(-1, *(() if x.ndim == 1 else x.shape[1:]))
                        self.data[k] = self._interp1d_torch(t_src, x, t_tgt)

            # update time + dt + mask
            self.data["time"] = t_tgt
            dt = torch.zeros_like(t_tgt)
            dt[1:] = t_tgt[1:] - t_tgt[:-1]
            dt[0] = dt[1]
            self.data["dt"] = dt
            self.data["mask"] = torch.ones(t_tgt.shape[0], dtype=torch.bool, device=self.device)
            
            print(f"✓ Resampled to {resample_hz} Hz: {len(t_src)} -> {len(t_tgt)} frames")
            
        self._compute_madgwick_orientations(beta=0.01)
        
        self._compute_mahony_orientations(kp=0.005, ki=0.0001)
        
        if 'mag_vec' in self.data:  # magnetometer-aided VQF only when the sequence carries mag data
            self.compute_vqf_orientation(use_mag=True, out_key="vqf_orientation_9d",
                                        vqf_params = dict(
                                        tauAcc=3,
                                        tauMag=100,
                                        magDistRejectionEnabled=True,
                                        magNormTh=0.1,
                                        magDipTh=10.0,
                                        magRejectionFactor=5,
                                    ))
        
        self.compute_vqf_orientation(use_mag=False, out_key="vqf_orientation_6d",
                            vqf_params = dict(
                                    tauAcc=3,
                                    tauMag=100,
                                    magDistRejectionEnabled=True,
                                    magNormTh=0.05,
                                    magDipTh=10.0,
                                    magRejectionFactor=5,
                                ))
        print(f"\n=== Loaded Nymeria dataset: {data_name} ===")

    def _compute_mahony_orientations(
        self,
        kp: float = 1.0,
        ki: float = 0.3,
        store_keys: bool = True,
    ):
        """
        Uses ahrs.filters.Mahony (not the custom torch implementation).

        Outputs (PyPose SO3, xyzw, CPF->World):
        - self.data["mahony_q6"], self.data["mahony_b6"] (if bias is exposed)
        - self.data["mahony_q9"], self.data["mahony_b9"] (if mag_vec exists)
        """
        import numpy as np
        import torch
        import pypose as pp
        from ahrs.filters import Mahony  # ahrs package :contentReference[oaicite:4]{index=4}

        assert self.coordinate == "body_coord", \
            "Mahony requires SENSOR-frame IMU. Use coordinate='body_coord'."
        assert "gyro" in self.data and "acc" in self.data and "dt" in self.data

        gyro_cpf = self.data["gyro"]
        acc_cpf  = self.data["acc"]
        dt_t     = self.data["dt"].reshape(-1)

        device = gyro_cpf.device
        dtype  = gyro_cpf.dtype

        # CPF -> Mahony/Madgwick frame (gravity +Z)
        gyro_mad_t = self._cpf_to_madgwick_frame(gyro_cpf)
        acc_mad_t  = self._cpf_to_madgwick_frame(acc_cpf)
        
        gyr = gyro_mad_t.detach().cpu().numpy().astype(np.float64)
        acc = acc_mad_t.detach().cpu().numpy().astype(np.float64)
        dt  = dt_t.detach().cpu().numpy().astype(np.float64)

        N = gyr.shape[0]
        q0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)  # wxyz

        # --- IMU (6D) ---
        mah = Mahony(k_P=float(kp), k_I=float(ki))  # k_P/k_I in docs :contentReference[oaicite:5]{index=5}
        Q6 = np.zeros((N, 4), dtype=np.float64)
        Q6[0] = q0

        # Optional bias trace (depends on ahrs version; keep it robust)
        B6 = np.zeros((N, 3), dtype=np.float64)

        for t in range(1, N):
            Q6[t] = mah.updateIMU(Q6[t-1], gyr=gyr[t], acc=acc[t], dt=float(dt[t]))
            if hasattr(mah, "b") and mah.b is not None:
                B6[t] = np.asarray(mah.b, dtype=np.float64).reshape(3)
            else:
                B6[t] = B6[t-1]

        Q6_wxyz_mad = torch.from_numpy(Q6).to(device=device, dtype=dtype)
        Q6_wxyz_cpf = self._madgwick_to_cpf_quat(Q6_wxyz_mad)

        # bias back to CPF (your inverse mapping)
        B6_mad_t = torch.from_numpy(B6).to(device=device, dtype=dtype)
        B6_cpf = torch.stack([B6_mad_t[:, 0], B6_mad_t[:, 2], -B6_mad_t[:, 1]], dim=-1)

        Q6_xyzw = torch.stack([Q6_wxyz_cpf[:, 1], Q6_wxyz_cpf[:, 2], Q6_wxyz_cpf[:, 3], Q6_wxyz_cpf[:, 0]], dim=-1)
        mahony_q6 = pp.SO3(Q6_xyzw)

        if store_keys:
            mahony_q6 = self._align_so3_to_gt_at_t0(mahony_q6, align_to_gt=True)
            self.data["mahony_q6"] = mahony_q6
            self.data["mahony_b6"] = B6_cpf

        # --- MARG (9D) if magnetometer exists ---
        mag_vec = self.data.get("mag_vec", None)
        if mag_vec is not None:
            mag_mad_t = self._cpf_to_madgwick_frame(mag_vec)
            mag = mag_mad_t.detach().cpu().numpy().astype(np.float64)
            mag = normalize_rows(mag)
                                 
            mah9 = Mahony(k_P=float(kp), k_I=float(ki))
            Q9 = np.zeros((N, 4), dtype=np.float64)
            Q9[0] = q0
            B9 = np.zeros((N, 3), dtype=np.float64)

            for t in range(1, N):
                Q9[t] = mah9.updateMARG(Q9[t-1], gyr=gyr[t], acc=acc[t], mag=mag[t], dt=float(dt[t]))
                if hasattr(mah9, "b") and mah9.b is not None:
                    B9[t] = np.asarray(mah9.b, dtype=np.float64).reshape(3)
                else:
                    B9[t] = B9[t-1]

            Q9_wxyz_mad = torch.from_numpy(Q9).to(device=device, dtype=dtype)
            Q9_wxyz_cpf = self._madgwick_to_cpf_quat(Q9_wxyz_mad)

            B9_mad_t = torch.from_numpy(B9).to(device=device, dtype=dtype)
            B9_cpf = torch.stack([B9_mad_t[:, 0], B9_mad_t[:, 2], -B9_mad_t[:, 1]], dim=-1) 

            Q9_xyzw = torch.stack([Q9_wxyz_cpf[:, 1], Q9_wxyz_cpf[:, 2], Q9_wxyz_cpf[:, 3], Q9_wxyz_cpf[:, 0]], dim=-1)
            mahony_q9 = pp.SO3(Q9_xyzw)

            if store_keys:
                mahony_q9 = self._align_so3_to_gt_at_t0(mahony_q9, align_to_gt=True)
                self.data["mahony_q9"] = mahony_q9
                self.data["mahony_b9"] = B9_cpf
    
    def _align_so3_to_gt_at_t0(
        self,
        so3_est: pp.SO3,
        align_to_gt: bool = True,
        gt_key: str = "gt_orientation",
    ):
        """
        Align an estimated orientation trajectory's world frame to GT world frame at t=0.

        We assume both represent CPF->World (i.e., rotate CPF vectors into some "world").
        If the estimator's "world" differs by a constant rotation, we compute:

            R_align = R_gt(0) @ R_est(0)^T
            R_est_aligned(t) = R_align @ R_est(t)

        Args:
            so3_est: pp.SO3 [N] estimated orientation (CPF->World_est)
            align_to_gt: if False, returns so3_est unchanged
            gt_key: key in self.data holding GT pp.SO3 [N] (CPF->World_gt)

        Returns:
            pp.SO3 [N] aligned estimate (CPF->World_gt)
        """
        import torch
        import pypose as pp

        if (not align_to_gt) or (gt_key not in self.data) or (self.data[gt_key] is None):
            return so3_est

        gt_orientation = self.data[gt_key]
        if not isinstance(gt_orientation, pp.LieTensor):
            # best-effort: if GT not pp.SO3, skip alignment
            return so3_est

        # rotation matrices
        R_est_0 = so3_est[0].matrix()          # [3,3]
        R_gt_0  = gt_orientation[0].matrix()   # [3,3]

        # constant alignment from est-world to gt-world
        R_align = R_gt_0 @ R_est_0.transpose(-1, -2)  # [3,3]

        # apply to all
        R_est_all = so3_est.matrix()  # [N,3,3]
        R_aligned = torch.matmul(R_align.unsqueeze(0), R_est_all)  # [N,3,3]

        return pp.mat2SO3(R_aligned)


    def compute_vqf_orientation(
        self,
        use_mag: bool = True,
        out_key: str = "vqf_orientation",
        store_bias: bool = True,
        vqf_params: dict = None,
        align_to_gt: bool = True,
    ):
        """
        Compute orientation using the vqf library and store it in self.data[out_key] as pp.SO3 (PyPose).
        
        - Inputs: self.data['gyro'] [N,3] rad/s, self.data['acc'] [N,3] m/s^2 (or normalized ok)
        - Optional: self.data['mag_vec'] [N,3] (mag vector) for 9D fusion
        
        Args:
            align_to_gt: If True and 'gt_orientation' exists, align VQF frame to GT world at t=0
        """
        # Lazy import so your dataset still loads even if vqf isn't installed
        from vqf import VQF  

        assert "gyro" in self.data and "acc" in self.data and "dt" in self.data, \
            "Need self.data['gyro'], self.data['acc'], self.data['dt']"

        gyro_t = self.data["gyro"]
        acc_t  = self.data["acc"]
        dt_t   = self.data["dt"].reshape(-1)

        device = gyro_t.device
        dtype  = gyro_t.dtype

        # Convert to numpy for vqf
        gyr = gyro_t.detach().cpu().numpy().astype(np.float64)
        acc = acc_t.detach().cpu().numpy().astype(np.float64)
        dt  = dt_t.detach().cpu().numpy().astype(np.float64)

        # Choose a single sampling time (VQF supports different rates, but simplest is one Ts)
        # Use median to be robust to occasional dt spikes
        # gyrTs = float(np.median(dt[1:])) if len(dt) > 1 else float(dt[0])
        gyrTs = 0.005  # Nymeria is 200 Hz

        params = vqf_params or {}
        vqf = VQF(gyrTs, **params)
        
        if use_mag:
            mag_vec = self.data.get("mag_vec", None)
            if mag_vec is None:
                raise ValueError(
                    "use_mag=True requires self.data['mag_vec'] as [N,3] magnetometer vectors."
                )
            mag = mag_vec.detach().cpu().numpy().astype(np.float64)

            # Use updateBatch which returns full trajectory
            out = vqf.updateBatch(gyr, acc, mag)
            # out['quat9D'] has shape [N, 4] in wxyz format
            Q_wxyz_SE = out['quat9D']  # [N, 4] wxyz
        else:
            out = vqf.updateBatch(gyr, acc)
            Q_wxyz_SE = out['quat6D']  # [N, 4] wxyz
            

        # Convert wxyz -> xyzw for PyPose
        Q_xyzw_SE = Q_wxyz_SE[:, [1, 2, 3, 0]].astype(np.float32)

        Q_xyzw_t = torch.from_numpy(Q_xyzw_SE).to(device=device, dtype=dtype)
        vqf_orientation = pp.SO3(Q_xyzw_t)

        # Align VQF "Earth" to GT "World" at t=0 if requested
        if align_to_gt and "gt_orientation" in self.data:
            gt_orientation = self.data["gt_orientation"]  # pp.SO3 [N, 4]
            
            # Get initial orientations
            R_vqf_0 = vqf_orientation[0].matrix()  # [3, 3] VQF at t=0
            R_gt_0 = gt_orientation[0].matrix()    # [3, 3] GT at t=0
            
            # Compute alignment: R_align = R_gt(0) @ R_vqf(0)^(-1)
            # This transforms VQF's Earth frame to GT's World frame
            R_align = R_gt_0 @ R_vqf_0.T  # [3, 3]
            
            # Apply to all VQF orientations: R_aligned(t) = R_align @ R_vqf(t)
            R_vqf_all = vqf_orientation.matrix()  # [N, 3, 3]
            R_aligned = torch.matmul(R_align.unsqueeze(0), R_vqf_all)  # [N, 3, 3]
            
            # Convert back to SO3
            vqf_orientation = pp.mat2SO3(R_aligned)
        
        self.data[out_key] = vqf_orientation

        if store_bias:
            # VQF has getBiasEstimate(); returns current bias + uncertainty (API varies slightly by version)
            try:
                bias_info = vqf.getBiasEstimate()
                self.data[out_key + "_gyro_bias"] = torch.tensor(
                    bias_info[0], device=device, dtype=dtype
                )
            except Exception:
                pass

        return self.data[out_key]
        
    def get_length(self):
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
                pass
            else:
                raise ValueError(f"Unsupported coordinate system: {coordinate}")
                
        except Exception as e:
            print(f"Error during coordinate transformation: {e}")
            raise e
    
    def _interp1d_torch(self, t_src, x_src, t_tgt):
        """
        Linear interpolation for torch tensors.
        t_src: [N]
        x_src: [N, ...]
        t_tgt: [M]
        returns: [M, ...]
        """
        import torch

        t_src = t_src.reshape(-1)
        t_tgt = t_tgt.reshape(-1)

        N = t_src.numel()
        M = t_tgt.numel()

        # indices on src for each target
        idx_r = torch.searchsorted(t_src, t_tgt, right=False)
        idx_r = torch.clamp(idx_r, 0, N - 1)
        idx_l = torch.clamp(idx_r - 1, 0, N - 1)

        t_l = t_src[idx_l]
        t_r = t_src[idx_r]
        denom = (t_r - t_l).clamp_min(1e-9)
        w = ((t_tgt - t_l) / denom).to(x_src.dtype)  # [M]

        # gather x_l, x_r
        x_l = x_src[idx_l]
        x_r = x_src[idx_r]

        # broadcast w to match x shape
        while w.ndim < x_l.ndim:
            w = w.unsqueeze(-1)

        return x_l + w * (x_r - x_l)

    def _resample_so3_slerp(self, R_src: pp.LieTensor, t_src: torch.Tensor, t_tgt: torch.Tensor):
        """
        Slerp orientations in pp.SO3 along time.
        R_src: [N] pp.SO3 (CPF→World)
        t_src: [N]
        t_tgt: [M]
        returns: [M] pp.SO3
        """
        import torch
        import pypose as pp

        t_src = t_src.reshape(-1)
        t_tgt = t_tgt.reshape(-1)
        N = t_src.numel()

        idx_r = torch.searchsorted(t_src, t_tgt, right=False)
        idx_r = torch.clamp(idx_r, 0, N - 1)
        idx_l = torch.clamp(idx_r - 1, 0, N - 1)

        t_l = t_src[idx_l]
        t_r = t_src[idx_r]
        denom = (t_r - t_l).clamp_min(1e-9)
        alpha = ((t_tgt - t_l) / denom).to(t_src.dtype)  # [M]

        Rl = R_src[idx_l]
        Rr = R_src[idx_r]

        # relative rotation from left to right
        Rrel = Rl.Inv() * Rr                      # [M]
        w = (Rrel.Log() * alpha.unsqueeze(-1))    # [M,3] in so3
        Rt = Rl * pp.so3(w).Exp()                 # [M] SO3
        return Rt

    @staticmethod
    def _q_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """
        Hamilton product for quaternions in wxyz.
        q1, q2: (..., 4)
        returns: (..., 4)
        """
        w1, x1, y1, z1 = q1.unbind(-1)
        w2, x2, y2, z2 = q2.unbind(-1)
        return torch.stack([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ], dim=-1)

    @staticmethod
    def _q_conj(q: torch.Tensor) -> torch.Tensor:
        """Conjugate of wxyz quaternion."""
        w, x, y, z = q.unbind(-1)
        return torch.stack([w, -x, -y, -z], dim=-1)

    @staticmethod
    def _q_norm(q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        """Normalize quaternion."""
        n = torch.linalg.norm(q, dim=-1, keepdim=True).clamp_min(eps)
        return q / n
    
    @staticmethod
    def _cpf_to_madgwick_frame(v: torch.Tensor) -> torch.Tensor:
        # Want: +Y_cpf -> +Z_mad
        x = v[..., 0:1]
        y = v[..., 1:2]
        z = v[..., 2:3]
        return torch.cat([x, -z, y], dim=-1)   # [x, -z, +y]
    
    @staticmethod
    def _madgwick_to_cpf_quat(q_wxyz: torch.Tensor):
        """
        Convert quaternion from Madgwick frame back to CPF frame.

        Given vector mapping: v_mad = C * v_cpf, where C = R_x(-90deg)
        Rotation matrices satisfy: R_cpf = C^T * R_mad * C
        => quaternions: q_cpf = qC^{-1} ⊗ q_mad ⊗ qC   (wxyz)
        """
        device, dtype = q_wxyz.device, q_wxyz.dtype
        sqrt2_2 = torch.tensor(0.7071067811865476, device=device, dtype=dtype)

        # C = R_x(+90deg)  => qC = [√2/2, +√2/2, 0, 0]
        qC     = torch.tensor([sqrt2_2,  sqrt2_2, 0.0, 0.0], device=device, dtype=dtype)  # cpf->mad
        qC_inv = torch.tensor([sqrt2_2, -sqrt2_2, 0.0, 0.0], device=device, dtype=dtype)  # mad->cpf

        def q_mul_wxyz(q1, q2):
            w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
            w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
            return torch.stack([
                w1*w2 - x1*x2 - y1*y2 - z1*z2,
                w1*x2 + x1*w2 + y1*z2 - z1*y2,
                w1*y2 - x1*z2 + y1*w2 + z1*x2,
                w1*z2 + x1*y2 - y1*x2 + z1*w2
            ], dim=-1)

        N = q_wxyz.shape[0]
        qC_b     = qC.unsqueeze(0).expand(N, -1)
        qC_inv_b = qC_inv.unsqueeze(0).expand(N, -1)

        # q_cpf = qC^{-1} * q_mad * qC
        q_temp = q_mul_wxyz(qC_inv_b, q_wxyz)
        q_cpf  = q_mul_wxyz(q_temp, qC_b)
        return q_cpf

    def _compute_madgwick_orientations(
        self,
        beta: float = 0.04,   # map to ahrs "gain"
        store_keys: bool = True,
    ):
        """
        Uses ahrs.filters.Madgwick (not the custom torch implementation).

        Outputs (PyPose SO3, xyzw, CPF->World):
        - self.data["ori_imu6"]
        - self.data["ori_marg9"] (if mag_vec exists)
        """
        import numpy as np
        import torch
        import pypose as pp
        from ahrs.filters import Madgwick  # ahrs package

        assert self.coordinate == "body_coord", \
            "Madgwick requires SENSOR-frame IMU. Use coordinate='body_coord'."
        assert "gyro" in self.data and "acc" in self.data and "dt" in self.data

        gyro_cpf = self.data["gyro"]
        acc_cpf  = self.data["acc"]
        dt_t     = self.data["dt"].reshape(-1)

        device = gyro_cpf.device
        dtype  = gyro_cpf.dtype

        # CPF -> Madgwick frame (gravity +Z)
        gyro_mad_t = self._cpf_to_madgwick_frame(gyro_cpf)
        acc_mad_t  = self._cpf_to_madgwick_frame(acc_cpf)

        # Torch -> numpy (double for stability)
        gyr = gyro_mad_t.detach().cpu().numpy().astype(np.float64)
        acc = acc_mad_t.detach().cpu().numpy().astype(np.float64)
        dt  = dt_t.detach().cpu().numpy().astype(np.float64)

        N = gyr.shape[0]
        q0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)  # wxyz

        # --- IMU (6D) ---
        mad = Madgwick(gain=float(beta))  # "gain" in ahrs docs :contentReference[oaicite:3]{index=3}
        Q6 = np.zeros((N, 4), dtype=np.float64)
        Q6[0] = q0
        for t in range(1, N):
            Q6[t] = mad.updateIMU(Q6[t-1], gyr=gyr[t], acc=acc[t], dt=float(dt[t]))

        # Back to torch, convert Mad-frame quaternion -> CPF quaternion
        Q6_wxyz_mad = torch.from_numpy(Q6).to(device=device, dtype=dtype)
        Q6_wxyz_cpf = self._madgwick_to_cpf_quat(Q6_wxyz_mad)

        # wxyz -> xyzw for PyPose
        Q6_xyzw = torch.stack([Q6_wxyz_cpf[:, 1], Q6_wxyz_cpf[:, 2], Q6_wxyz_cpf[:, 3], Q6_wxyz_cpf[:, 0]], dim=-1)
        ori_imu6 = pp.SO3(Q6_xyzw)
        ori_imu6 = self._align_so3_to_gt_at_t0(ori_imu6, align_to_gt=True)
        if store_keys:
            self.data["ori_imu6"] = ori_imu6

        # --- MARG (9D) if magnetometer exists ---
        mag_vec = self.data.get("mag_vec", None)
        if mag_vec is not None:
            mag_mad_t = self._cpf_to_madgwick_frame(mag_vec)
            mag = mag_mad_t.detach().cpu().numpy().astype(np.float64)
            mag = normalize_rows(mag)
            
            mad9 = Madgwick(gain=float(beta))
            Q9 = np.zeros((N, 4), dtype=np.float64)
            Q9[0] = q0
            for t in range(1, N):
                Q9[t] = mad9.updateMARG(Q9[t-1], gyr=gyr[t], acc=acc[t], mag=mag[t], dt=float(dt[t]))

            Q9_wxyz_mad = torch.from_numpy(Q9).to(device=device, dtype=dtype)
            Q9_wxyz_cpf = self._madgwick_to_cpf_quat(Q9_wxyz_mad)

            Q9_xyzw = torch.stack([Q9_wxyz_cpf[:, 1], Q9_wxyz_cpf[:, 2], Q9_wxyz_cpf[:, 3], Q9_wxyz_cpf[:, 0]], dim=-1)
            ori_marg9 = pp.SO3(Q9_xyzw)
            ori_marg9 = self._align_so3_to_gt_at_t0(ori_marg9, align_to_gt=True)
            if store_keys:
                self.data["ori_marg9"] = ori_marg9

        
    def plot_imu_timeseries(self, out_path: str = "imu_timeseries.png"):
        """
        Plot self.data["acc"] and self.data["gyro"] as time series (x,y,z)
        in a 2x3 grid: row 0 = acc (m/s^2), row 1 = gyro (rad/s).
        Saves the figure to `out_path` (PNG).
        """
        import numpy as np
        import torch
        import matplotlib.pyplot as plt

        # --- Fetch data ---
        acc  = self.data.get("acc", None)
        gyro = self.data.get("gyro", None)
        if acc is None or gyro is None:
            raise ValueError("Both 'acc' and 'gyro' must exist in self.data to plot.")

        # Torch → numpy (CPU)
        def to_np(t):
            if isinstance(t, torch.Tensor):
                t = t.detach().cpu()
            return np.asarray(t)

        acc_np  = to_np(acc)
        gyro_np = to_np(gyro)

        if acc_np.ndim != 2 or acc_np.shape[1] != 3:
            raise ValueError(f"'acc' must be [N,3], got {acc_np.shape}")
        if gyro_np.ndim != 2 or gyro_np.shape[1] != 3:
            raise ValueError(f"'gyro' must be [N,3], got {gyro_np.shape}")

        N = acc_np.shape[0]

        # --- Time vector ---
        if "time" in self.data and self.data["time"] is not None:
            t = to_np(self.data["time"]).reshape(-1)
            if t.shape[0] != N:
                raise ValueError(f"Time length {t.shape[0]} != data length {N}.")
        elif "dt" in self.data and self.data["dt"] is not None:
            dt = to_np(self.data["dt"]).reshape(-1)
            if dt.shape[0] != N:
                # if dt is N-1, accumulate and prepend the first value
                if dt.shape[0] == N - 1:
                    t = np.concatenate([[0.0], np.cumsum(dt)])
                else:
                    raise ValueError(f"dt length {dt.shape[0]} incompatible with data length {N}.")
            else:
                t = np.cumsum(dt)
                t -= t[0]
        else:
            # fallback to sample index (seconds) if no time/dt available
            t = np.arange(N, dtype=float)

        # --- Plot ---
        comp_labels = ["x", "y", "z"]
        fig, axes = plt.subplots(2, 3, figsize=(15, 6), sharex=True)
        fig.suptitle("IMU Time Series", fontsize=14)

        # Acc row
        for i in range(3):
            ax = axes[0, i]
            ax.plot(t, acc_np[:, i], linewidth=1.0)
            ax.set_title(f"acc-{comp_labels[i]} (m/s²)")
            ax.grid(True, alpha=0.3)

        # Gyro row
        for i in range(3):
            ax = axes[1, i]
            ax.plot(t, gyro_np[:, i], linewidth=1.0)
            ax.set_title(f"gyro-{comp_labels[i]} (rad/s)")
            ax.grid(True, alpha=0.3)

        axes[1, 1].set_xlabel("time (s)")

        plt.tight_layout(rect=[0, 0.02, 1, 0.95])
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"✓ Saved IMU time series to: {out_path}")
        
    def plot_mag_yaw_vs_gt_yaw(
        self,
        save_path: str = None,   # e.g., "data_vis/mag_yaw_vs_gt_yaw.png"
        show: bool = True,
        units: str = "deg",      # "deg" or "rad"
        unwrap: bool = True,     # unroll 2π jumps for cleaner lines
        title: str = "Magnetometer yaw vs. GT yaw",
        align_start: bool = True # NEW: shift mag so both series start equal
    ):
        import numpy as np
        import torch
        import matplotlib.pyplot as plt

        if "mag" not in self.data:
            raise KeyError("'mag' not found. Compute it before calling this.")
        if "gt_orientation" not in self.data:
            raise KeyError("'gt_orientation' not found in dataset.")

        def to_np(x):
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()
            return np.asarray(x)

        def nearest_align(t_ref, t_src, X_src):
            t_ref = np.asarray(t_ref, dtype=float)
            t_src = np.asarray(t_src, dtype=float)
            X_src = np.asarray(X_src)
            idx_r = np.searchsorted(t_src, t_ref, side="left")
            idx_r = np.clip(idx_r, 0, t_src.size - 1)
            idx_l = np.clip(idx_r - 1, 0, t_src.size - 1)
            take_r = np.abs(t_src[idx_r] - t_ref) <= np.abs(t_src[idx_l] - t_ref)
            nn = np.where(take_r, idx_r, idx_l)
            return X_src[nn]

        # Times
        t_pose = self.data.get("time", None)
        t_pose = to_np(t_pose) if t_pose is not None else np.arange(len(self.data["gt_orientation"]), dtype=float)
        t_imu  = self._get_time_imu_np() if hasattr(self, "_get_time_imu_np") else None
        if t_imu is None:
            t_imu = np.arange(len(self.data["mag"]), dtype=float)

        # GT yaw from orientation (radians)
        so3 = self.data["gt_orientation"]
        Rw = so3.matrix() if hasattr(so3, "matrix") else torch.as_tensor(so3)
        Rw = Rw.float()
        fwd = Rw[:, :, 2]                      # [N,3], body +Z in world
        fwd = to_np(fwd)
        yaw_gt = np.arctan2(fwd[:, 0], fwd[:, 2])  # rad

        n_pose = min(len(t_pose), yaw_gt.shape[0])
        t_pose = t_pose[:n_pose]
        yaw_gt = yaw_gt[:n_pose]

        # Mag yaw (already radians in self.data['mag']) aligned to pose-rate
        yaw_mag = to_np(self.data["mag"]).reshape(-1)
        yaw_mag_aligned = nearest_align(t_pose, t_imu, yaw_mag)

        # Optional unwrap (radians)
        if unwrap:
            yaw_gt = np.unwrap(yaw_gt)
            yaw_mag_aligned = np.unwrap(yaw_mag_aligned)

        # NEW: align starts (do in radians for robustness)
        if align_start and yaw_gt.size and yaw_mag_aligned.size:
            if unwrap:
                offset = yaw_gt[0] - yaw_mag_aligned[0]
            else:
                # circular smallest-difference offset in [-pi, pi]
                offset = np.arctan2(np.sin(yaw_gt[0] - yaw_mag_aligned[0]),
                                    np.cos(yaw_gt[0] - yaw_mag_aligned[0]))
            yaw_mag_aligned = yaw_mag_aligned + offset
            if not unwrap:
                # keep within [-pi, pi] if you didn't unwrap
                yaw_mag_aligned = np.arctan2(np.sin(yaw_mag_aligned), np.cos(yaw_mag_aligned))

        # Units
        if units.lower() == "deg":
            yaw_gt_plot = np.degrees(yaw_gt)
            yaw_mag_plot = np.degrees(yaw_mag_aligned)
            ylab = "Yaw (deg)"
            err = yaw_mag_plot - yaw_gt_plot
        else:
            yaw_gt_plot = yaw_gt
            yaw_mag_plot = yaw_mag_aligned
            ylab = "Yaw (rad)"
            err = yaw_mag_plot - yaw_gt_plot

        diff = np.arctan2(np.sin(yaw_mag_aligned - yaw_gt),
                  np.cos(yaw_mag_aligned - yaw_gt))   # rad in [-pi,pi]
        rmse = np.degrees(np.sqrt(np.mean(diff**2)))

        # Plot (top: signals, bottom: |Δ|)
        fig, ax = plt.subplots(1, 1, figsize=(10, 4), sharex=True)
        ax.plot(t_pose, yaw_gt_plot, label="GT yaw", linewidth=1.8, color="#0088FF")
        ax.plot(t_pose, yaw_mag_plot, label="Mag yaw ", linewidth=1.6, linestyle="--", color="#FF8C00")
        ax.set_ylabel(ylab, fontsize=14)
        ax.grid(True, linewidth=0.5, alpha=0.5)
        ax.legend()
        ax.set_title(f"{title}", fontsize=16)

        # ax[1].plot(t_pose, np.abs(err), linewidth=1.4)
        ax.set_xlabel("Time (s)", fontsize=14)
        # ax.set_xlim(0, 1000)
        # ax[1].set_ylabel(f"|Δ| ({units})")
        ax.grid(True, linewidth=0.5, alpha=0.5)

        fig.tight_layout()
        if save_path:
            import os
            dirpath = os.path.dirname(save_path)
            if dirpath:
                os.makedirs(dirpath, exist_ok=True)
            fig.savefig(save_path, dpi=200, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)

        print(f"[INFO] mag vs gt_yaw: N={len(t_pose)}, RMSE={rmse:.4f} {units}, align_start={align_start}")

    def plot_mag_yaw_error(
        self,
        save_path: str = "data_vis/mag_yaw_error.png",
        show: bool = False,
        units: str = "deg",      # "deg" or "rad"
        xlim: tuple = None,
        title: str = "Magnetometer Yaw Error vs. GT"
    ):
        """
        Plot the angular error between magnetometer yaw and ground truth yaw.
        
        Computes angular difference accounting for wrap-around at ±π.
        Shows error over time with statistics (mean, std, RMSE, max).
        
        Args:
            save_path: Path to save the PNG plot
            show: Whether to display the plot
            units: "deg" for degrees or "rad" for radians
            xlim: Tuple (xmin, xmax) for x-axis limits, or None for auto
            title: Title for the plot
        """
        import numpy as np
        import torch
        import matplotlib.pyplot as plt
        import os

        if "mag" not in self.data:
            raise KeyError("'mag' not found. Compute it before calling this.")
        if "gt_orientation" not in self.data:
            raise KeyError("'gt_orientation' not found in dataset.")

        def to_np(x):
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()
            return np.asarray(x)

        def nearest_align(t_ref, t_src, X_src):
            t_ref = np.asarray(t_ref, dtype=float)
            t_src = np.asarray(t_src, dtype=float)
            X_src = np.asarray(X_src)
            idx_r = np.searchsorted(t_src, t_ref, side="left")
            idx_r = np.clip(idx_r, 0, t_src.size - 1)
            idx_l = np.clip(idx_r - 1, 0, t_src.size - 1)
            take_r = np.abs(t_src[idx_r] - t_ref) <= np.abs(t_src[idx_l] - t_ref)
            nn = np.where(take_r, idx_r, idx_l)
            return X_src[nn]

        def angular_difference(angle1, angle2):
            """
            Compute the shortest angular difference between two angles.
            Handles wrap-around at ±π.
            
            Returns: difference in range [-π, π] (radians)
            """
            diff = angle1 - angle2
            # Wrap to [-π, π]
            diff = np.arctan2(np.sin(diff), np.cos(diff))
            return diff

        # Get time vectors
        t_pose = self.data.get("time", None)
        t_pose = to_np(t_pose) if t_pose is not None else np.arange(len(self.data["gt_orientation"]), dtype=float)
        t_imu = self._get_time_imu_np() if hasattr(self, "_get_time_imu_np") else None
        if t_imu is None:
            t_imu = np.arange(len(self.data["mag"]), dtype=float)

        # GT yaw from orientation (radians)
        so3 = self.data["gt_orientation"]
        Rw = so3.matrix() if hasattr(so3, "matrix") else torch.as_tensor(so3)
        Rw = Rw.float()
        fwd = Rw[:, :, 2]                      # [N,3], body +Z in world
        fwd = to_np(fwd)
        yaw_gt = np.arctan2(fwd[:, 0], fwd[:, 2])  # rad

        n_pose = min(len(t_pose), yaw_gt.shape[0])
        t_pose = t_pose[:n_pose]
        yaw_gt = yaw_gt[:n_pose]

        # Mag yaw (already radians in self.data['mag']) aligned to pose-rate
        yaw_mag = to_np(self.data["mag"]).reshape(-1)
        # yaw_mag_aligned = nearest_align(t_pose, t_imu, yaw_mag)
        yaw_mag_aligned = yaw_mag 

        offset0 = angular_difference(yaw_gt[0], yaw_mag_aligned[0])  # wrap(yaw_gt0 - yaw_mag0)
        yaw_mag_aligned = np.arctan2(
            np.sin(yaw_mag_aligned + offset0),
            np.cos(yaw_mag_aligned + offset0)
        )

        # Compute angular error (radians, in range [-π, π])
        yaw_error = angular_difference(yaw_mag_aligned, yaw_gt)

        # Convert to degrees if requested
        if units.lower() == "deg":
            yaw_error_plot = np.degrees(yaw_error)
            ylabel = "Yaw Error (deg)"
        else:
            yaw_error_plot = yaw_error
            ylabel = "Yaw Error (rad)"

        # Compute statistics
        mean_error = np.mean(yaw_error_plot)
        std_error = np.std(yaw_error_plot)
        max_error = np.max(np.abs(yaw_error_plot))
        rmse = np.sqrt(np.mean(yaw_error_plot ** 2))

        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 5))

        # Plot yaw error
        ax.plot(t_pose, yaw_error_plot,
               color='#FF8C00',
               linewidth=1.5,
               alpha=0.85,
               label=f'Mag Yaw Error (Mean: {mean_error:.2f}, Std: {std_error:.2f}, Max: {max_error:.2f})')

        # Add zero reference line
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)

        # Configure plot
        ax.set_ylabel(ylabel, fontsize=13, fontweight='bold')
        ax.set_xlabel("Time (s)", fontsize=13, fontweight='bold')
        ax.grid(True, linewidth=0.5, alpha=0.4)
        ax.legend(loc='upper right', fontsize=10)

        if xlim is not None:
            ax.set_xlim(xlim)

        # Set title
        fig.suptitle(title, fontsize=16, fontweight='bold')
        fig.tight_layout()

        # Save figure
        dirpath = os.path.dirname(save_path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"[INFO] Saved: {save_path} (RMSE: {rmse:.4f} {units})")

        if show:
            plt.show()
        else:
            plt.close(fig)

    def plot_altitude_vs_gt_y(
            self,
            save_path: str = None,
            show: bool = True,
            match: str = "offset",   # "none" | "offset" | "affine"
            invert_alt: bool = False,# True if your altitude sign is opposite to +Y up
            title: str = "Barometric Altitude vs. GT Altitude",
            dpi: int = 200,
        ):
            """
            Compare gt_translation Y (world) vs barometric altitude.

            - Aligns IMU-rate altitude to pose-rate GT timestamps via nearest neighbor.
            - 'match' controls how altitude is brought to the GT scale:
                * 'none'   : raw altitude (after optional invert) plotted against GT
                * 'offset' : shift altitude so alt_adj[0] == gt_y[0]
                * 'affine' : solve y ≈ s*alt + b (least-squares) and plot s*alt + b
            - Set invert_alt=True if baro altitude increases opposite to your +Y axis.

            Saves a 2-row figure: (1) signals overlay, (2) absolute error |Δ|.
            """
            import numpy as np, torch, matplotlib.pyplot as plt

            # --- checks ---
            if "gt_translation" not in self.data:
                raise KeyError("gt_translation not found in self.data")
            if "altitude" not in self.data:
                raise KeyError("altitude not found in self.data")

            # --- helpers ---
            def to_np(x):
                if isinstance(x, torch.Tensor):
                    return x.detach().cpu().numpy()
                return np.asarray(x)

            def get_time_pose_np():
                t = self.data.get("time_pose", self.data.get("time", None))
                return to_np(t) if t is not None else None

            def get_time_imu_np():
                # Prefer explicit IMU time; else reconstruct from dt_imu; else None
                t = self.data.get("time_imu", None)
                if t is not None: return to_np(t)
                if "dt_imu" in self.data:
                    dt = to_np(self.data["dt_imu"])
                    if dt.size:
                        t = dt.cumsum() - dt[0]
                        return t
                return None

            def nearest_align(t_ref, t_src, X_src):
                t_ref = np.asarray(t_ref, dtype=float)
                t_src = np.asarray(t_src, dtype=float)
                X_src = np.asarray(X_src)
                idx_r = np.searchsorted(t_src, t_ref, side="left")
                idx_r = np.clip(idx_r, 0, t_src.size - 1)
                idx_l = np.clip(idx_r - 1, 0, t_src.size - 1)
                take_r = np.abs(t_src[idx_r] - t_ref) <= np.abs(t_src[idx_l] - t_ref)
                nn = np.where(take_r, idx_r, idx_l)
                return X_src[nn]

            # --- fetch series ---
            gt = to_np(self.data["gt_translation"])  # [N,3]
            if gt.ndim != 2 or gt.shape[1] != 3:
                raise ValueError(f"gt_translation must be [N,3], got {gt.shape}")
            gt_y = gt[:, 1]  # world Y

            alt = to_np(self.data["altitude"]).reshape(-1)  # [M]
            if invert_alt:
                alt = -alt

            # --- time vectors & alignment ---
            t_pose = get_time_pose_np()
            if t_pose is None:
                # Fallback: index-as-time at pose rate
                t_pose = np.arange(gt_y.shape[0], dtype=float)
            if t_pose.shape[0] != gt_y.shape[0]:
                n = min(t_pose.shape[0], gt_y.shape[0])
                t_pose = t_pose[:n]
                gt_y   = gt_y[:n]

            t_imu = get_time_imu_np()
            if t_imu is None:
                # If we truly have no IMU time, truncate to pose length
                n = min(alt.shape[0], gt_y.shape[0])
                alt = alt[:n]
                t_aligned = t_pose[:n]
                gt_y = gt_y[:n]
            else:
                # Align altitude to pose timestamps
                alt = nearest_align(t_pose, t_imu, alt)
                t_aligned = t_pose

            # --- bring altitude onto GT Y scale ---
            alt_adj = alt.copy()
            if match == "offset":
                # Make the first samples equal (simple bias correction)
                if alt_adj.size > 0:
                    alt_adj = alt_adj + (gt_y[0] - alt_adj[0])
            elif match == "affine":
                # Solve least-squares y ≈ s*alt + b
                # [alt 1] * [s b]^T ≈ y
                A = np.stack([alt_adj, np.ones_like(alt_adj)], axis=1)
                x, *_ = np.linalg.lstsq(A, gt_y, rcond=None)
                s, b = x
                alt_adj = s * alt_adj + b
            elif match == "none":
                pass
            else:
                raise ValueError("match must be one of {'none','offset','affine'}")

            # --- error ---
            n = min(gt_y.shape[0], alt_adj.shape[0])
            gt_y = gt_y[:n]; alt_adj = alt_adj[:n]; t_aligned = t_aligned[:n]
            err = np.abs(alt_adj - gt_y)

            # --- plot ---
            fig, ax = plt.subplots(1, 1, figsize=(10, 4), sharex=True)
            # ax.plot(t_aligned, gt_y, label="GT Height", linewidth=2.6)
            # ax.plot(t_aligned, alt_adj, label=f"Barometric Altitude", linewidth=2.2, linestyle="--")
            ax.plot(t_aligned, gt_y,    label="GT Altitude",
                    linewidth=3.6, color="#0088FF", zorder=3)        # blue
            ax.plot(t_aligned, alt_adj, label="Barometric Altitude",
                    linewidth=3.2, color="#FF8C00", linestyle="--", zorder=2)  # red
            ax.set_ylabel("Altitude (m)", fontsize=14)
            ax.grid(True, linewidth=0.5, alpha=0.5)
            ax.legend()

            # ax[1].plot(t_aligned, err)
            ax.set_xlabel("Time (s)", fontsize=14)
            ax.set_xlim(0, 1000)
            # ax[1].set_ylabel("|Δ| (m)")
            # ax[1].grid(True, linewidth=0.5, alpha=0.5)
            fig.suptitle(title, fontsize=16)
            fig.tight_layout()

            if save_path:
                fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
            if show:
                plt.show()
            else:
                plt.close(fig)
    
    def _get_time_imu_np(self):
        """Return IMU-rate time vector as numpy (tries time_imu, then dt_imu, else index)."""
        import numpy as np, torch
        def to_np(x):
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()
            return np.asarray(x)
        t = None
        if getattr(self, "is_dual_rate_data", False):
            t = self.data.get("time_imu", None)
            if t is None and "dt_imu" in self.data:
                dt = to_np(self.data["dt_imu"])
                if dt.size:
                    t = dt.cumsum() - dt[0]
        else:
            t = self.data.get("time", None)
        if t is None:
            # will be constructed per series length by caller
            return None
        return to_np(t)
    
    def plot_orientation_comparison(
        self,
        save_dir: str = "data_vis",
        show: bool = False,
        dpi: int = 200,
        xlim: tuple = None,
    ):
        """
        Plot quaternion components comparing GT with each estimation method separately.
        
        Creates 4 separate PNG files:
        - GT vs Gyro-integrated orientation
        - GT vs VQF orientation (9-axis)
        - GT vs Madgwick orientation (9-axis)
        - GT vs Mahony orientation (9-axis)
        
        Args:
            save_dir: Directory to save the PNG plots
            show: Whether to display the plots
            dpi: DPI for saved figures
            xlim: Tuple (xmin, xmax) for x-axis limits, or None for auto
        """
        import numpy as np
        import torch
        import matplotlib.pyplot as plt
        import os
        
        def to_np(x):
            """Convert tensor to numpy array."""
            if isinstance(x, torch.Tensor):
                if hasattr(x, 'tensor'):  # PyPose SO3
                    return x.tensor().detach().cpu().numpy()
                return x.detach().cpu().numpy()
            return np.asarray(x)
        
        # Get time vector
        t = self.data.get("time", None)
        if t is None:
            raise ValueError("Time data not found in self.data")
        t = to_np(t)
        
        # Get GT orientation
        if 'gt_orientation' not in self.data or self.data['gt_orientation'] is None:
            print("[ERROR] gt_orientation not found in self.data")
            return
        
        gt_ori = to_np(self.data['gt_orientation'])
        if gt_ori.ndim == 1:
            gt_ori = gt_ori.reshape(-1, 4)
        
        # Define comparison pairs: (key, label, color, filename)
        comparisons = [
            ('orientation_from_gyro', 'Gyro', '#FF8C00', 'gt_vs_gyro.png'),
            ('vqf_orientation_9d', 'VQF-9D', '#FF0000', 'gt_vs_vqf_9d.png'),
            ('vqf_orientation_6d', 'VQF-6D', '#FF6666', 'gt_vs_vqf_6d.png'),
            ('ori_marg9', 'Madgwick-9D', '#00CC66', 'gt_vs_madgwick_9d.png'),
            ('ori_imu6', 'Madgwick-6D', '#66FF99', 'gt_vs_madgwick_6d.png'),
            ('mahony_q9', 'Mahony-9D', '#CC00CC', 'gt_vs_mahony_9d.png'),
            ('mahony_q6', 'Mahony-6D', '#FF66FF', 'gt_vs_mahony_6d.png'),
        ]
        
        component_names = ['qx', 'qy', 'qz', 'qw']  # PyPose uses xyzw format
        
        # Create output directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Generate each comparison plot
        for ori_key, method_label, method_color, filename in comparisons:
            if ori_key not in self.data or self.data[ori_key] is None:
                print(f"[WARNING] {ori_key} not found in self.data, skipping {filename}...")
                continue
            
            # Get method orientation data
            method_ori = to_np(self.data[ori_key])
            
            # Handle shape: should be [N, 4] for quaternions
            if method_ori.ndim == 1:
                method_ori = method_ori.reshape(-1, 4)
            elif method_ori.ndim != 2 or method_ori.shape[1] != 4:
                print(f"[WARNING] {ori_key} has unexpected shape {method_ori.shape}, skipping...")
                continue
            
            # Ensure time and orientation have same length
            n = min(len(t), len(gt_ori), len(method_ori))
            t_plot = t[:n]
            gt_plot = gt_ori[:n]
            method_plot = method_ori[:n]
            
            # Create figure with 4 subplots (one for each quaternion component)
            fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
            
            # Plot each quaternion component
            for i, (ax, comp_name) in enumerate(zip(axes, component_names)):
                # GT line
                ax.plot(t_plot, gt_plot[:, i], 
                       label='GT', 
                       color='#0088FF', 
                       linestyle='-', 
                       linewidth=2.5,
                       alpha=0.85)
                
                # Method line
                ax.plot(t_plot, method_plot[:, i], 
                       label=method_label, 
                       color=method_color, 
                       linestyle='--', 
                       linewidth=2.0,
                       alpha=0.85)
                
                # Configure subplot
                ax.set_ylabel(comp_name, fontsize=13, fontweight='bold')
                ax.grid(True, linewidth=0.5, alpha=0.4)
                ax.set_ylim(-1.1, 1.1)
                
                if i == 0:
                    ax.legend(loc='upper right', fontsize=11)
            
            # Set x-axis label and limits
            axes[-1].set_xlabel("Time (s)", fontsize=13, fontweight='bold')
            if xlim is not None:
                axes[-1].set_xlim(xlim)
            
            # Set title
            fig.suptitle(f"GT vs {method_label} Orientation", fontsize=16, fontweight='bold')
            fig.tight_layout()
            
            # Save figure
            save_path = os.path.join(save_dir, filename)
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
            print(f"[INFO] Saved: {save_path}")
            
            if show:
                plt.show()
            else:
                plt.close(fig)
    
    def plot_euler_comparison(
        self,
        save_dir: str = "data_vis",
        show: bool = False,
        dpi: int = 200,
        xlim: tuple = None,
        units: str = "deg",  # "deg" or "rad"
        unwrap: bool = True,  # unwrap angles to remove ±π jumps
    ):
        """
        Plot Euler angles (Yaw, Pitch, Roll) comparing GT with each estimation method separately.
        
        Creates 4 separate PNG files:
        - GT vs Gyro-integrated orientation (Euler angles)
        - GT vs VQF orientation (9-axis, Euler angles)
        - GT vs Madgwick orientation (9-axis, Euler angles)
        - GT vs Mahony orientation (9-axis, Euler angles)
        
        Args:
            save_dir: Directory to save the PNG plots
            show: Whether to display the plots
            dpi: DPI for saved figures
            xlim: Tuple (xmin, xmax) for x-axis limits, or None for auto
            units: "deg" for degrees or "rad" for radians
            unwrap: If True, unwrap angles to remove ±π jumps for cleaner visualization
        """
        import numpy as np
        import torch
        import matplotlib.pyplot as plt
        import os
        
        def to_np(x):
            """Convert tensor to numpy array."""
            if isinstance(x, torch.Tensor):
                if hasattr(x, 'tensor'):  # PyPose SO3
                    return x.tensor().detach().cpu().numpy()
                return x.detach().cpu().numpy()
            return np.asarray(x)
        
        def quat_to_euler(q):
            """
            Convert quaternion (xyzw format) to Euler angles (roll, pitch, yaw).
            
            Args:
                q: quaternion array [N, 4] in xyzw format (qx, qy, qz, qw)
            
            Returns:
                euler: [N, 3] array with (roll, pitch, yaw) in radians
            """
            qx, qy, qz, qw = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
            
            # Roll (x-axis rotation)
            sinr_cosp = 2 * (qw * qx + qy * qz)
            cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
            roll = np.arctan2(sinr_cosp, cosr_cosp)
            
            # Pitch (y-axis rotation)
            sinp = 2 * (qw * qy - qz * qx)
            sinp = np.clip(sinp, -1.0, 1.0)
            pitch = np.arcsin(sinp)
            
            # Yaw (z-axis rotation)
            siny_cosp = 2 * (qw * qz + qx * qy)
            cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
            yaw = np.arctan2(siny_cosp, cosy_cosp)
            
            return np.stack([roll, pitch, yaw], axis=1)
        
        # Get time vector
        t = self.data.get("time", None)
        if t is None:
            raise ValueError("Time data not found in self.data")
        t = to_np(t)
        
        # Get GT orientation
        if 'gt_orientation' not in self.data or self.data['gt_orientation'] is None:
            print("[ERROR] gt_orientation not found in self.data")
            return
        
        gt_quat = to_np(self.data['gt_orientation'])
        if gt_quat.ndim == 1:
            gt_quat = gt_quat.reshape(-1, 4)
        
        # Convert GT to Euler angles
        gt_euler = quat_to_euler(gt_quat)  # [N, 3]: roll, pitch, yaw
        
        # Define comparison pairs: (key, label, color, filename)
        comparisons = [
            ('orientation_from_gyro', 'Gyro', '#FF8C00', 'gt_vs_gyro_euler.png'),
            ('vqf_orientation_9d', 'VQF-9D', '#FF0000', 'gt_vs_vqf_9d_euler.png'),
            ('vqf_orientation_6d', 'VQF-6D', '#FF6666', 'gt_vs_vqf_6d_euler.png'),
            ('ori_marg9', 'Madgwick-9D', '#00CC66', 'gt_vs_madgwick_9d_euler.png'),
            ('ori_imu6', 'Madgwick-6D', '#66FF99', 'gt_vs_madgwick_6d_euler.png'),
            ('mahony_q9', 'Mahony-9D', '#CC00CC', 'gt_vs_mahony_9d_euler.png'),
            ('mahony_q6', 'Mahony-6D', '#FF66FF', 'gt_vs_mahony_6d_euler.png'),
        ]
        
        euler_names = ['Roll', 'Pitch', 'Yaw']
        
        # Create output directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Generate each comparison plot
        for ori_key, method_label, method_color, filename in comparisons:
            if ori_key not in self.data or self.data[ori_key] is None:
                print(f"[WARNING] {ori_key} not found in self.data, skipping {filename}...")
                continue
            
            # Get method orientation data
            method_quat = to_np(self.data[ori_key])
            
            # Handle shape: should be [N, 4] for quaternions
            if method_quat.ndim == 1:
                method_quat = method_quat.reshape(-1, 4)
            elif method_quat.ndim != 2 or method_quat.shape[1] != 4:
                print(f"[WARNING] {ori_key} has unexpected shape {method_quat.shape}, skipping...")
                continue
            
            # Convert method to Euler angles
            method_euler = quat_to_euler(method_quat)  # [N, 3]: roll, pitch, yaw
            
            # Ensure time and orientation have same length
            n = min(len(t), len(gt_euler), len(method_euler))
            t_plot = t[:n]
            gt_plot = gt_euler[:n].copy()
            method_plot = method_euler[:n].copy()
            
            # Optional unwrap (in radians) to remove ±π jumps
            if unwrap:
                for i in range(3):  # unwrap each angle (roll, pitch, yaw) separately
                    gt_plot[:, i] = np.unwrap(gt_plot[:, i])
                    method_plot[:, i] = np.unwrap(method_plot[:, i])
            
            # Convert to degrees if requested
            if units.lower() == "deg":
                gt_plot = np.degrees(gt_plot)
                method_plot = np.degrees(method_plot)
                ylabel_unit = "(deg)"
            else:
                ylabel_unit = "(rad)"
            
            # Create figure with 3 subplots (one for each Euler angle)
            fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
            
            # Plot each Euler angle
            for i, (ax, euler_name) in enumerate(zip(axes, euler_names)):
                # GT line
                ax.plot(t_plot, gt_plot[:, i], 
                       label='GT', 
                       color='#0088FF', 
                       linestyle='-', 
                       linewidth=2.5,
                       alpha=0.85)
                
                # Method line
                ax.plot(t_plot, method_plot[:, i], 
                       label=method_label, 
                       color=method_color, 
                       linestyle='--', 
                       linewidth=2.0,
                       alpha=0.85)
                
                # Configure subplot
                ax.set_ylabel(f"{euler_name} {ylabel_unit}", fontsize=13, fontweight='bold')
                ax.grid(True, linewidth=0.5, alpha=0.4)
                
                if i == 0:
                    ax.legend(loc='upper right', fontsize=11)
            
            # Set x-axis label and limits
            axes[-1].set_xlabel("Time (s)", fontsize=13, fontweight='bold')
            if xlim is not None:
                axes[-1].set_xlim(xlim)
            
            # Set title
            fig.suptitle(f"GT vs {method_label} Euler Angles", fontsize=16, fontweight='bold')
            fig.tight_layout()
            
            # Save figure
            save_path = os.path.join(save_dir, filename)
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
            print(f"[INFO] Saved: {save_path}")
            
            if show:
                plt.show()
            else:
                plt.close(fig)
    
    def plot_angle_error_comparison(
        self,
        save_dir: str = "data_vis",
        show: bool = False,
        dpi: int = 200,
        xlim: tuple = None,
        units: str = "deg",  # "deg" or "rad"
    ):
        """
        Plot geodesic angle error between GT and each estimation method separately.
        
        The geodesic angle error is computed as:
        1. Compute relative quaternion: q_err = q_gt ⊗ q_method^(-1)
        2. Extract angle: θ = 2 * arccos(|w_err|)
        
        This gives the cleanest 1D curve showing orientation error over time.
        
        Creates 4 separate PNG files:
        - GT vs Gyro angle error
        - GT vs VQF angle error (9-axis)
        - GT vs Madgwick angle error (9-axis)
        - GT vs Mahony angle error (9-axis)
        
        Args:
            save_dir: Directory to save the PNG plots
            show: Whether to display the plots
            dpi: DPI for saved figures
            xlim: Tuple (xmin, xmax) for x-axis limits, or None for auto
            units: "deg" for degrees or "rad" for radians
        """
        import numpy as np
        import torch
        import matplotlib.pyplot as plt
        import os
        
        def to_np(x):
            """Convert tensor to numpy array."""
            if isinstance(x, torch.Tensor):
                if hasattr(x, 'tensor'):  # PyPose SO3
                    return x.tensor().detach().cpu().numpy()
                return x.detach().cpu().numpy()
            return np.asarray(x)
        
        def quat_inverse(q):
            """
            Compute quaternion inverse (conjugate for unit quaternions).
            
            Args:
                q: quaternion array [N, 4] in xyzw format (qx, qy, qz, qw)
            
            Returns:
                q_inv: [N, 4] inverse quaternion
            """
            # For unit quaternions, inverse = conjugate
            qx, qy, qz, qw = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
            return np.stack([-qx, -qy, -qz, qw], axis=1)
        
        def quat_multiply(q1, q2):
            """
            Multiply two quaternions: q1 ⊗ q2
            
            Args:
                q1, q2: quaternion arrays [N, 4] in xyzw format (qx, qy, qz, qw)
            
            Returns:
                q_result: [N, 4] product quaternion
            """
            x1, y1, z1, w1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
            x2, y2, z2, w2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
            
            w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
            x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
            y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
            z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
            
            return np.stack([x, y, z, w], axis=1)
        
        def geodesic_angle_error(q_gt, q_est):
            """
            Compute geodesic angle error between two quaternions.
            
            Args:
                q_gt: ground truth quaternion [N, 4] in xyzw format
                q_est: estimated quaternion [N, 4] in xyzw format
            
            Returns:
                theta: [N] angle error in radians, range [0, π]
            """
            # Compute relative rotation: q_err = q_gt ⊗ q_est^(-1)
            q_est_inv = quat_inverse(q_est)
            q_err = quat_multiply(q_gt, q_est_inv)
            
            # Extract w component and compute angle
            w_err = q_err[:, 3]  # w is the 4th component in xyzw format
            
            # Clamp to [-1, 1] for numerical stability
            w_err = np.clip(np.abs(w_err), 0.0, 1.0)
            
            # θ = 2 * arccos(|w_err|)
            theta = 2.0 * np.arccos(w_err)
            
            return theta
        
        # Get time vector
        t = self.data.get("time", None)
        if t is None:
            raise ValueError("Time data not found in self.data")
        t = to_np(t)
        
        # Get GT orientation
        if 'gt_orientation' not in self.data or self.data['gt_orientation'] is None:
            print("[ERROR] gt_orientation not found in self.data")
            return
        
        gt_quat = to_np(self.data['gt_orientation'])
        if gt_quat.ndim == 1:
            gt_quat = gt_quat.reshape(-1, 4)
        
        # Define comparison pairs: (key, label, color, filename)
        comparisons = [
            ('orientation_from_gyro', 'Gyro', '#FF8C00', 'angle_error_gyro.png'),
            ('vqf_orientation_9d', 'VQF-9D', '#FF0000', 'angle_error_vqf_9d.png'),
            ('vqf_orientation_6d', 'VQF-6D', '#FF6666', 'angle_error_vqf_6d.png'),
            ('ori_marg9', 'Madgwick-9D', '#00CC66', 'angle_error_madgwick_9d.png'),
            ('ori_imu6', 'Madgwick-6D', '#66FF99', 'angle_error_madgwick_6d.png'),
            ('mahony_q9', 'Mahony-9D', '#CC00CC', 'angle_error_mahony_9d.png'),
            ('mahony_q6', 'Mahony-6D', '#FF66FF', 'angle_error_mahony_6d.png'),
        ]
        
        # Create output directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Generate each comparison plot
        for ori_key, method_label, method_color, filename in comparisons:
            if ori_key not in self.data or self.data[ori_key] is None:
                print(f"[WARNING] {ori_key} not found in self.data, skipping {filename}...")
                continue
            
            # Get method orientation data
            method_quat = to_np(self.data[ori_key])
            
            # Handle shape: should be [N, 4] for quaternions
            if method_quat.ndim == 1:
                method_quat = method_quat.reshape(-1, 4)
            elif method_quat.ndim != 2 or method_quat.shape[1] != 4:
                print(f"[WARNING] {ori_key} has unexpected shape {method_quat.shape}, skipping...")
                continue
            
            # Ensure time and orientation have same length
            n = min(len(t), len(gt_quat), len(method_quat))
            t_plot = t[:n]
            gt_plot = gt_quat[:n]
            method_plot = method_quat[:n]
            
            # Compute geodesic angle error
            angle_error = geodesic_angle_error(gt_plot, method_plot)  # radians
            
            # Convert to degrees if requested
            if units.lower() == "deg":
                angle_error_plot = np.degrees(angle_error)
                ylabel = "Angle Error (deg)"
            else:
                angle_error_plot = angle_error
                ylabel = "Angle Error (rad)"
            
            # Compute statistics
            mean_error = np.mean(angle_error_plot)
            std_error = np.std(angle_error_plot)
            max_error = np.max(angle_error_plot)
            rmse = np.sqrt(np.mean(angle_error_plot ** 2))
            
            # Create figure
            fig, ax = plt.subplots(1, 1, figsize=(12, 5))
            
            # Plot angle error
            ax.plot(t_plot, angle_error_plot, 
                   color=method_color, 
                   linewidth=1.5,
                   alpha=0.85,
                   label=f'{method_label} (Mean: {mean_error:.2f}, Std: {std_error:.2f}, Max: {max_error:.2f})')
            
            # Configure plot
            ax.set_ylabel(ylabel, fontsize=13, fontweight='bold')
            ax.set_xlabel("Time (s)", fontsize=13, fontweight='bold')
            ax.grid(True, linewidth=0.5, alpha=0.4)
            ax.legend(loc='upper right', fontsize=10)
            ax.set_ylim(bottom=0)  # Angle error is always >= 0
            
            if xlim is not None:
                ax.set_xlim(xlim)
            
            # Set title
            fig.suptitle(f"Geodesic Angle Error: GT vs {method_label}", fontsize=16, fontweight='bold')
            fig.tight_layout()
            
            # Save figure
            save_path = os.path.join(save_dir, filename)
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
            print(f"[INFO] Saved: {save_path} (RMSE: {rmse:.4f} {units})")
            
            if show:
                plt.show()
            else:
                plt.close(fig)
    
    def plot_magnetometer_stability(
        self,
        save_path: str = "data_vis/mag_stability.png",
        show: bool = False,
        dpi: int = 200,
        xlim: tuple = None,
    ):
        """
        Plot magnetometer magnitude and dip angle to check stability.
        
        Computes and plots:
        1. Magnitude: m = ||mag_vec||
        2. Dip angle: dip = -arcsin(m_z / ||m||)
        
        These metrics help assess magnetometer data quality and stability over time.
        Ideally, both should be relatively constant if there's no magnetic disturbance.
        
        Args:
            save_path: Path to save the PNG plot
            show: Whether to display the plot
            dpi: DPI for saved figure
            xlim: Tuple (xmin, xmax) for x-axis limits, or None for auto
        """
        import numpy as np
        import torch
        import matplotlib.pyplot as plt
        import os
        
        def to_np(x):
            """Convert tensor to numpy array."""
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()
            return np.asarray(x)
        
        # Check if mag_vec exists
        if 'mag_vec' not in self.data or self.data['mag_vec'] is None:
            print("[ERROR] mag_vec not found in self.data. Cannot plot magnetometer stability.")
            return
        
        # Get time vector
        t = self.data.get("time", None)
        if t is None:
            raise ValueError("Time data not found in self.data")
        t = to_np(t)
        
        # Get magnetometer vector data
        mag_vec = to_np(self.data['mag_vec'])  # [N, 3]
        
        if mag_vec.ndim != 2 or mag_vec.shape[1] != 3:
            print(f"[ERROR] mag_vec has unexpected shape {mag_vec.shape}, expected [N, 3]")
            return
        
        # Ensure time and mag_vec have same length
        n = min(len(t), len(mag_vec))
        t_plot = t[:n]
        mag_vec = mag_vec[:n]
        
        # Compute magnitude: m = ||mag_vec||
        mag_magnitude = np.linalg.norm(mag_vec, axis=1)  # [N]
        
        # Compute dip angle: dip = -arcsin(m_z / ||m||)
        # m_z is the z-component (index 2)
        m_z = mag_vec[:, 2]
        # Clip to avoid numerical issues with arcsin
        ratio = np.clip(m_z / (mag_magnitude + 1e-12), -1.0, 1.0)
        mag_dip = -np.arcsin(ratio)  # radians
        mag_dip_deg = np.degrees(mag_dip)  # convert to degrees
        
        # Compute statistics
        mag_mean = np.mean(mag_magnitude)
        mag_std = np.std(mag_magnitude)
        dip_mean = np.mean(mag_dip_deg)
        dip_std = np.std(mag_dip_deg)
        
        # Create figure with 2 subplots
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Plot 1: Magnetometer Magnitude
        axes[0].plot(t_plot, mag_magnitude, 
                    color='#0088FF', 
                    linewidth=1.5,
                    alpha=0.85,
                    label=f'Mean: {mag_mean:.2f}, Std: {mag_std:.2f}')
        axes[0].axhline(y=mag_mean, color='red', linestyle='--', linewidth=1.0, alpha=0.7, label='Mean')
        axes[0].set_ylabel('Magnitude ||m|| (μT)', fontsize=13, fontweight='bold')
        axes[0].grid(True, linewidth=0.5, alpha=0.4)
        axes[0].legend(loc='upper right', fontsize=10)
        axes[0].set_title('Magnetometer Magnitude', fontsize=14, fontweight='bold')
        
        # Plot 2: Magnetic Dip Angle
        axes[1].plot(t_plot, mag_dip_deg, 
                    color='#FF8C00', 
                    linewidth=1.5,
                    alpha=0.85,
                    label=f'Mean: {dip_mean:.2f}°, Std: {dip_std:.2f}°')
        axes[1].axhline(y=dip_mean, color='red', linestyle='--', linewidth=1.0, alpha=0.7, label='Mean')
        axes[1].set_ylabel('Dip Angle (deg)', fontsize=13, fontweight='bold')
        axes[1].set_xlabel('Time (s)', fontsize=13, fontweight='bold')
        axes[1].grid(True, linewidth=0.5, alpha=0.4)
        axes[1].legend(loc='upper right', fontsize=10)
        axes[1].set_title('Magnetic Dip Angle: -arcsin(m_z / ||m||)', fontsize=14, fontweight='bold')
        
        if xlim is not None:
            axes[1].set_xlim(xlim)
        
        # Set overall title
        fig.suptitle('Magnetometer Stability Check', fontsize=16, fontweight='bold')
        fig.tight_layout()
        
        # Save figure
        dirpath = os.path.dirname(save_path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"[INFO] Saved magnetometer stability plot: {save_path}")
        print(f"       Magnitude - Mean: {mag_mean:.2f} μT, Std: {mag_std:.2f} μT")
        print(f"       Dip Angle - Mean: {dip_mean:.2f}°, Std: {dip_std:.2f}°")
        
        if show:
            plt.show()
        else:
            plt.close(fig)

    def plot_acceleration_mean(
        self,
        save_path: str = "data_vis/acc_mean.png",
        show: bool = False,
        dpi: int = 200,
        xlim: tuple = None,
        window_size: int = 200,  # Window size for rolling mean (in samples)
        plot_raw: bool = True,   # Whether to plot raw acceleration as well
    ):
        """
        Plot acceleration data and its rolling mean for X, Y, Z axes.
        
        Computes and plots:
        1. Raw acceleration signals (optional, semi-transparent)
        2. Rolling mean of acceleration for each axis
        3. Overall mean as horizontal lines
        
        Args:
            save_path: Path to save the PNG plot
            show: Whether to display the plot
            dpi: DPI for saved figure
            xlim: Tuple (xmin, xmax) for x-axis limits, or None for auto
            window_size: Window size for computing rolling mean (in samples)
            plot_raw: Whether to plot raw acceleration (semi-transparent)
        """
        import os
        import matplotlib.pyplot as plt
        import numpy as np
        
        def to_np(x):
            """Convert tensor to numpy array."""
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()
            return np.asarray(x)
        
        # Check if acc exists
        if 'acc' not in self.data or self.data['acc'] is None:
            print("[ERROR] acc not found in self.data. Cannot plot acceleration.")
            return
        
        # Get time vector
        t = self.data.get("time", None)
        if t is None:
            raise ValueError("Time data not found in self.data")
        t = to_np(t)
        
        # Get acceleration data
        acc = to_np(self.data['acc'])  # [N, 3]
        
        if acc.ndim != 2 or acc.shape[1] != 3:
            print(f"[ERROR] acc has unexpected shape {acc.shape}, expected [N, 3]")
            return
        
        # Ensure time and acc have same length
        n = min(len(t), len(acc))
        t_plot = t[:n]
        acc = acc[:n]
        
        # Extract X, Y, Z components
        acc_x = acc[:, 0]
        acc_y = acc[:, 1]
        acc_z = acc[:, 2]
        
        # Compute rolling mean using convolution
        def rolling_mean(data, window):
            kernel = np.ones(window) / window
            # Use 'same' mode to keep same length, handle edges
            return np.convolve(data, kernel, mode='same')
        
        acc_x_mean = rolling_mean(acc_x, window_size)
        acc_y_mean = rolling_mean(acc_y, window_size)
        acc_z_mean = rolling_mean(acc_z, window_size)
        
        # Compute overall statistics
        stats = {
            'X': {'mean': np.mean(acc_x), 'std': np.std(acc_x)},
            'Y': {'mean': np.mean(acc_y), 'std': np.std(acc_y)},
            'Z': {'mean': np.mean(acc_z), 'std': np.std(acc_z)},
        }
        
        # Colors for each axis
        colors = {'X': '#FF4444', 'Y': '#44AA44', 'Z': '#4444FF'}
        
        # Create figure with 3 subplots
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        
        axis_data = [
            ('X', acc_x, acc_x_mean),
            ('Y', acc_y, acc_y_mean),
            ('Z', acc_z, acc_z_mean),
        ]
        
        for ax, (axis_name, raw, mean) in zip(axes, axis_data):
            color = colors[axis_name]
            s = stats[axis_name]
            
            # Plot raw acceleration (semi-transparent)
            if plot_raw:
                ax.plot(t_plot, raw, 
                       color=color, 
                       linewidth=0.5,
                       alpha=0.3,
                       label='Raw')
            
            # Plot rolling mean
            ax.plot(t_plot, mean, 
                   color=color, 
                   linewidth=2.0,
                   alpha=0.9,
                   label=f'Rolling Mean (w={window_size})')
            
            # Plot overall mean as horizontal line
            ax.axhline(y=s['mean'], color='black', linestyle='--', 
                      linewidth=1.5, alpha=0.7, 
                      label=f"Overall Mean: {s['mean']:.3f}")
            
            # Configure subplot
            ax.set_ylabel(f'Acc {axis_name} (m/s²)', fontsize=13, fontweight='bold')
            ax.grid(True, linewidth=0.5, alpha=0.4)
            ax.legend(loc='upper right', fontsize=9)
            ax.set_title(f'Acceleration {axis_name}-axis (Mean: {s["mean"]:.3f}, Std: {s["std"]:.3f})', 
                        fontsize=12, fontweight='bold')
        
        # Set x-axis label and limits
        axes[-1].set_xlabel('Time (s)', fontsize=13, fontweight='bold')
        if xlim is not None:
            axes[-1].set_xlim(xlim)
        
        # Set overall title
        fig.suptitle('Acceleration Mean Analysis', fontsize=16, fontweight='bold')
        fig.tight_layout()
        
        # Save figure
        dirpath = os.path.dirname(save_path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"[INFO] Saved acceleration mean plot: {save_path}")
        print(f"       Acc X - Mean: {stats['X']['mean']:.4f} m/s², Std: {stats['X']['std']:.4f} m/s²")
        print(f"       Acc Y - Mean: {stats['Y']['mean']:.4f} m/s², Std: {stats['Y']['std']:.4f} m/s²")
        print(f"       Acc Z - Mean: {stats['Z']['mean']:.4f} m/s², Std: {stats['Z']['std']:.4f} m/s²")
        
        if show:
            plt.show()
        else:
            plt.close(fig)


    def print_error_per_minute(self):
        """Print mean geodesic error (deg) for each method, averaged over the full sequence."""
        import numpy as np

        def to_np_xyzw(key):
            v = self.data.get(key)
            if v is None:
                return None
            if isinstance(v, pp.LieTensor):
                return v.tensor().detach().cpu().numpy()
            if isinstance(v, torch.Tensor):
                return v.detach().cpu().numpy()
            return np.asarray(v)

        def geodesic_deg(q_gt, q_est):
            q_gt  = q_gt  / (np.linalg.norm(q_gt,  axis=1, keepdims=True) + 1e-9)
            q_est = q_est / (np.linalg.norm(q_est, axis=1, keepdims=True) + 1e-9)
            dot = np.clip(np.abs(np.einsum('ni,ni->n', q_gt, q_est)), 0.0, 1.0)
            return np.degrees(2.0 * np.arccos(dot))

        if 'gt_orientation' not in self.data:
            print("[ERROR] gt_orientation not found")
            return

        q_gt = to_np_xyzw('gt_orientation')

        methods = [
            ('orientation_from_gyro',      'Right Gyro '),
            ('orientation_from_gyro_left', 'Left Gyro  '),
            ('ori_imu6',                   'Madgwick-6D'),
            ('ori_marg9',                  'Madgwick-9D'),
            ('mahony_q6',                  'Mahony-6D  '),
            ('mahony_q9',                  'Mahony-9D  '),
        ]

        print("\n=== Mean geodesic error (deg) ===")
        for key, name in methods:
            q_est = to_np_xyzw(key)
            if q_est is None:
                print(f"  {name}: (not available)")
                continue
            n = min(len(q_gt), len(q_est))
            mean_err = geodesic_deg(q_gt[:n], q_est[:n]).mean()
            print(f"  {name}: {mean_err:.4f} deg")
        print("=================================")

    def plot_error_per_minute(
        self,
        save_path: str = "data_vis/error_per_minute.png",
        show: bool = False,
        dpi: int = 200,
    ):
        """
        Plot mean geodesic orientation error (degrees) per minute for six methods:
        Right Gyro, Left Gyro, Madgwick-6D, Madgwick-9D, Mahony-6D, Mahony-9D.

        Bins the per-frame geodesic error into 1-minute windows using self.data['time'],
        then plots mean error per bin as a bar/line chart.
        """
        import os
        import numpy as np
        import matplotlib.pyplot as plt

        def to_np_xyzw(key):
            v = self.data.get(key)
            if v is None:
                return None
            if isinstance(v, pp.LieTensor):
                return v.tensor().detach().cpu().numpy()   # [N, 4] xyzw
            if isinstance(v, torch.Tensor):
                return v.detach().cpu().numpy()
            return np.asarray(v)

        def geodesic_deg(q_gt, q_est):
            """xyzw inputs -> geodesic angle in degrees [N]"""
            # normalise
            q_gt  = q_gt  / (np.linalg.norm(q_gt,  axis=1, keepdims=True) + 1e-9)
            q_est = q_est / (np.linalg.norm(q_est, axis=1, keepdims=True) + 1e-9)
            dot = np.abs(np.einsum('ni,ni->n', q_gt, q_est))
            dot = np.clip(dot, 0.0, 1.0)
            return np.degrees(2.0 * np.arccos(dot))

        if 'gt_orientation' not in self.data:
            print("[ERROR] gt_orientation not found")
            return

        q_gt = to_np_xyzw('gt_orientation')
        t    = self.data['time'].detach().cpu().numpy().reshape(-1)
        t0   = t[0]
        t_min = (t - t0) / 60.0          # seconds -> minutes

        methods = [
            ('orientation_from_gyro',      'Right Gyro',  '#E74C3C'),
            ('orientation_from_gyro_left', 'Left Gyro',   '#3498DB'),
            ('ori_imu6',                   'Madgwick-6D', '#2ECC71'),
            ('ori_marg9',                  'Madgwick-9D', '#1ABC9C'),
            ('mahony_q6',                  'Mahony-6D',   '#9B59B6'),
            ('mahony_q9',                  'Mahony-9D',   '#F39C12'),
        ]

        # 1-minute bin edges
        max_min  = t_min[-1]
        bin_edges = np.arange(0, max_min + 1.0, 1.0)
        bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        bin_ids = np.digitize(t_min, bin_edges) - 1   # 0-based bin index
        n_bins  = len(bin_centres)

        fig, ax = plt.subplots(figsize=(max(8, n_bins * 0.6 + 2), 5))

        bar_width  = 0.8 / len(methods)
        x_pos      = np.arange(n_bins, dtype=float)

        for i, (key, label, color) in enumerate(methods):
            q_est = to_np_xyzw(key)
            if q_est is None:
                print(f"[WARNING] {key} not found, skipping")
                continue

            n = min(len(q_gt), len(q_est), len(t_min))
            err = geodesic_deg(q_gt[:n], q_est[:n])
            ids = bin_ids[:n]

            means = np.array([
                err[ids == b].mean() if np.any(ids == b) else np.nan
                for b in range(n_bins)
            ])

            offset = (i - (len(methods) - 1) / 2.0) * bar_width
            ax.bar(x_pos + offset, means, width=bar_width * 0.9,
                   color=color, label=label, alpha=0.85)

        ax.set_xlabel("Time (minutes)", fontsize=12, fontweight='bold')
        ax.set_ylabel("Mean geodesic error (deg)", fontsize=12, fontweight='bold')
        ax.set_title("Orientation error per minute", fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f"{int(b)}-{int(b)+1}" for b in bin_edges[:-1]], rotation=45, ha='right')
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(axis='y', linewidth=0.5, alpha=0.4)

        fig.tight_layout()
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"[INFO] Saved: {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

    def plot_gyro_integration_comparison(
        self,
        save_dir: str = "data_vis",
        show: bool = False,
        dpi: int = 200,
        xlim: tuple = None,
        units: str = "deg",
        unwrap: bool = True,
    ):
        """
        Compare orientation from right-IMU and left-IMU gyro integration against GT.

        Saves two PNGs to save_dir:
          - gyro_right_vs_gt.png  (GT vs Right Gyro)
          - gyro_left_vs_gt.png   (GT vs Left Gyro)
        Each has 3 subplots: Roll, Pitch, Yaw.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        import os

        def to_np(x):
            if isinstance(x, torch.Tensor):
                if hasattr(x, 'tensor'):
                    return x.tensor().detach().cpu().numpy()
                return x.detach().cpu().numpy()
            return np.asarray(x)

        def quat_to_euler(q):
            """xyzw -> (roll, pitch, yaw) in radians."""
            qx, qy, qz, qw = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
            roll  = np.arctan2(2*(qw*qx + qy*qz), 1 - 2*(qx*qx + qy*qy))
            sinp  = np.clip(2*(qw*qy - qz*qx), -1.0, 1.0)
            pitch = np.arcsin(sinp)
            yaw   = np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))
            return np.stack([roll, pitch, yaw], axis=1)

        if 'gt_orientation' not in self.data:
            print("[ERROR] gt_orientation not found in self.data")
            return

        t = to_np(self.data['time'])
        scale = np.degrees(1.0) if units == "deg" else 1.0
        ylabel_unit = "deg" if units == "deg" else "rad"
        angle_names = ["Roll", "Pitch", "Yaw"]
        os.makedirs(save_dir, exist_ok=True)

        def geodesic_mean_error(q_gt, q_est):
            """Mean geodesic angle error in degrees; q in xyzw format."""
            x1, y1, z1, w1 = q_gt[:, 0], q_gt[:, 1], q_gt[:, 2], q_gt[:, 3]
            x2, y2, z2, w2 = -q_est[:, 0], -q_est[:, 1], -q_est[:, 2], q_est[:, 3]
            w_err = w1*w2 - x1*x2 - y1*y2 - z1*z2
            theta = np.degrees(2.0 * np.arccos(np.clip(np.abs(w_err), 0.0, 1.0)))
            return float(np.mean(theta)), float(np.std(theta))

        # Pre-compute euler and raw quaternions for all three orientations
        euler_cache = {}
        quat_cache  = {}
        for key in ('gt_orientation', 'orientation_from_gyro', 'orientation_from_gyro_left'):
            if key in self.data and self.data[key] is not None:
                q = to_np(self.data[key])
                if q.ndim == 1:
                    q = q.reshape(-1, 4)
                quat_cache[key]  = q
                euler_cache[key] = quat_to_euler(q)

        if 'gt_orientation' not in euler_cache:
            print("[ERROR] gt_orientation missing euler conversion")
            return

        plots = [
            ('orientation_from_gyro',      'Right Gyro', '#FF8C00', 'gyro_right_vs_gt.png'),
            ('orientation_from_gyro_left', 'Left Gyro',  '#00CC66', 'gyro_left_vs_gt.png'),
        ]

        for gyro_key, gyro_label, gyro_color, filename in plots:
            if gyro_key not in euler_cache:
                print(f"[WARNING] {gyro_key} not found, skipping {filename}")
                continue

            n = min(len(t), len(euler_cache['gt_orientation']), len(euler_cache[gyro_key]))
            t_plot = t[:n]

            # Print per-angle mean absolute error and geodesic mean error
            angle_names_short = ["Roll", "Pitch", "Yaw"]
            print(f"\n[{gyro_label}] mean absolute error vs GT:")
            for angle_idx, angle_name in enumerate(angle_names_short):
                gt_vals  = np.unwrap(euler_cache['gt_orientation'][:n, angle_idx])
                est_vals = np.unwrap(euler_cache[gyro_key][:n, angle_idx])
                mae = np.mean(np.abs(np.degrees(gt_vals - est_vals)))
                print(f"  {angle_name:5s}: {mae:.4f} deg")
            mean_geo, std_geo = geodesic_mean_error(quat_cache['gt_orientation'][:n],
                                                    quat_cache[gyro_key][:n])
            print(f"  Geodesic: mean={mean_geo:.4f} deg, std={std_geo:.4f} deg")

            fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

            for ax, angle_idx, angle_name in zip(axes, range(3), angle_names):
                for key, label, color, ls, lw in [
                    ('gt_orientation', 'GT',       '#0088FF', '-',  2.5),
                    (gyro_key,         gyro_label, gyro_color, '--', 1.8),
                ]:
                    vals = euler_cache[key][:n, angle_idx].copy()
                    if unwrap:
                        vals = np.unwrap(vals)
                    ax.plot(t_plot, vals * scale, label=label, color=color,
                            linestyle=ls, linewidth=lw, alpha=0.85)
                ax.set_ylabel(f"{angle_name} ({ylabel_unit})", fontsize=12, fontweight='bold')
                ax.grid(True, linewidth=0.5, alpha=0.4)
                ax.legend(loc='upper right', fontsize=10)

            axes[-1].set_xlabel("Time (s)", fontsize=12, fontweight='bold')
            if xlim is not None:
                axes[-1].set_xlim(xlim)

            fig.suptitle(f"GT vs {gyro_label} – Roll / Pitch / Yaw", fontsize=15, fontweight='bold')
            fig.tight_layout()

            save_path = os.path.join(save_dir, filename)
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
            print(f"[INFO] Saved: {save_path}")

            if show:
                plt.show()
            else:
                plt.close(fig)


if __name__ == "__main__":
    data_root=os.path.join(os.environ.get("DATA_ROOT", "."), "processed_nymeria_2imu_both/test")
    data_name= "20231215_s1_dylan_lambert_act1_6bhj0h_cpfbody_200hz"
    # baro: 20231220_s1_amanda_wong_act6_z92lyz_cpfbody_200hz
    # baro: 20231211_s1_seth_bowman_act4_9jyykj_cpfbody_200hz
    
    # mag: 20231215_s1_dylan_lambert_act1_6bhj0h_cpfbody_200hz
    # mag: 20231220_s0_victor_sloan_act3_u9trb4_cpfbody_200hz
    # mag: 20231211_s1_seth_bowman_act4_9jyykj_cpfbody_200hz
    
    # data_root="$DATA_ROOT/processed_data_body_mag_baro_2imu_200hz_all/val"
    # data_name="20231020_s1_steven_jackson_act1_tdwliv_cpfbody_imu200hz_pose200hz"
    
    coordinate="body_coord"

    dataset = nymeria(
        data_root=data_root,
        data_name=data_name,
        coordinate=coordinate,
        remove_g=False,
        maximum_length=6000000000,
        rot_from_gyro=False,
        load_baro=True,
        load_left_imu=True,
        load_mag=True
    )
    
    # Print mean geodesic error per minute for all six methods
    dataset.print_error_per_minute()

    # Plot mean geodesic error per minute for all six methods
    dataset.plot_error_per_minute(
        save_path="data_vis/error_per_minute.png",
        show=False,
    )

    # Compare right vs left gyro integration against GT (two separate PNGs)
    dataset.plot_gyro_integration_comparison(
        save_dir="data_vis",
        show=False,
        units="deg",
        xlim=None,
    )
    
    # Plot Euler angle comparisons (creates 4 separate PNG files)
    dataset.plot_euler_comparison(
        save_dir="data_vis",
        show=False,
        xlim=None,
        units="deg",  # or "rad" for radians
        unwrap=True   # <<< IMPORTANT: removes ±π jumps for cleaner visualization
    )
    
    # # Plot geodesic angle error comparisons (creates 4 separate PNG files)
    dataset.plot_angle_error_comparison(
        save_dir="data_vis",
        show=False,
        xlim=None,
        units="deg"  # or "rad" for radians
    )
    
    # # Plot magnetometer stability check
    # dataset.plot_magnetometer_stability(
    #     save_path="data_vis/mag_stability.png",
    #     show=False,
    #     xlim=None
    # )
    
    # dataset.plot_altitude_vs_gt_y(
    #     save_path="data_vis/altitude_vs_gt_y.png",
    #     show=False,
    #     match="offset",         # try "affine" for scale+offset fit
    #     invert_alt=False        # set True if your baro sign is flipped
    # )
    # dataset.plot_mag_yaw_vs_gt_yaw(
    #     save_path="data_vis/mag_yaw_vs_gt_yaw.png",
    #     show=False,
    #     units="rad",
    #     unwrap=True
    # )
    
    # dataset.plot_mag_yaw_vs_gt_yaw(
    #     save_path="data_vis/mag_yaw_vs_gt_yaw.png",
    #     show=True,     # or False if you only want to save
    #     units="deg",   # degrees are easier to interpret
    #     unwrap=True    # <<< IMPORTANT: removes ±π jumps
    # )
    
    # # Plot magnetometer yaw error (geodesic angular difference)
    # dataset.plot_mag_yaw_error(
    #     save_path="data_vis/mag_yaw_error.png",
    #     show=True,     # or False if you only want to save
    #     units="deg",   # degrees are easier to interpret
    #     xlim=None
    # )
    # dataset.plot_acceleration_mean(
    #     save_path="data_vis/acc_mean.png",
    #     show=False,
    #     xlim=None,
    #     window_size=200,  # Rolling mean window (samples)
    #     plot_raw=True     # Show raw acc as semi-transparent background
    # )
    
# python -m datasets.nymeriadataset


