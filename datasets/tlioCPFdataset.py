import os
import json
import gc
import ctypes
import pickle
import numpy as np
import torch
import torch.nn.functional as F
import pypose as pp
from scipy.spatial.transform import Rotation as SciRot
import matplotlib.pyplot as plt

from .dataset import Sequence
from .dataset_utils import truncate_and_free

class tlioCPF(Sequence):
    """
    TLIO-style sequences loader with a 'nymeria'-like interface.

    Each item is a sliding window sampled from TLIO memmapped sequences.

    self.data (built at init) aggregates ALL samples across kept sequences:
        time:            [N,1]   microseconds
        gyro:            [N,3]   (imu gyroscope)
        acc:             [N,3]   (imu accelerometer)
        velocity:        [N,3]
        gt_translation:  [N,3]
        gt_orientation:  [N,4]   quaternion [x,y,z,w]
    """

    COMBINED_SENSOR_NAME = "combined"  # aligned style basename

    def __init__(
        self,
        data_root: str,
        data_name: str,
        coordinate: str = "glob_coord",
        maximum_length: int = 500000000,
        remove_g: bool = False,
        rot_from_gyro: bool = False,
        imu_in_body_frame: bool = False,
        **kwargs
    ):
        super().__init__()
        self.data_root = data_root
        self.data_name = data_name
        self.coordinate = coordinate

        # Handle _cpf suffix: add it only if not already present
        if not self.data_name.endswith('_cpf'):
            filename = f'{self.data_name}_cpf.npz'
        else:
            filename = f'{self.data_name}.npz'
        path = os.path.join(self.data_root, filename)

        with np.load(path, allow_pickle=False) as data:
            timestamps = torch.from_numpy(data["timestamps"]).float()
            positions = torch.from_numpy(data["positions"]).float()
            velocities = torch.from_numpy(data["velocities"]).float()
            orientations = torch.from_numpy(data["orientations"]).float()
            acc_body = torch.from_numpy(data["acc_body"]).float()
            gyro_body = torch.from_numpy(data["gyro_body"]).float()
        self.data = {}
        self.data["time"] = timestamps
        dt = torch.empty_like(timestamps)
        dt[1:] = timestamps[1:] - timestamps[:-1]
        dt[0] = dt[1]
        self.data["dt"] = dt
        self.data["acc"] = acc_body
        self.data["gyro"] = gyro_body
        self.data["velocity"] = velocities
        self.data["gt_translation"] = positions
        self.data['gt_orientation'] = pp.SO3(orientations)

        N = timestamps.shape[0]
        self.data["acc_left"]  = torch.zeros(N, 3, dtype=acc_body.dtype)
        self.data["gyro_left"] = torch.zeros(N, 3, dtype=gyro_body.dtype)
        self.data["mag"]       = torch.zeros(N, 1, dtype=acc_body.dtype)
        self.data["baro"]      = torch.zeros(N, 1, dtype=acc_body.dtype)
        self.data["mask"]      = torch.ones(N, 1, dtype=acc_body.dtype)
        
        if self.coordinate == "body_coord":
            self.data["velocity"] = self.data["gt_orientation"].Inv() @ self.data["velocity"]

        if remove_g:
            g_vec = np.array([0, 9.81, 0], dtype=np.float32)
            g_vec = torch.from_numpy(g_vec)
            g_vec_body = self.data["gt_orientation"].Inv() @ g_vec
            self.data["acc"] = self.data["acc"] - g_vec_body
        
        ##################### Rotation from gyro integration #############################
        if rot_from_gyro:
            # assume: import torch, import pypose as pp
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
            
        ###################################################################################

        # Rotate IMU to world frame only if coordinate is glob_coord AND imu_in_body_frame is False
        # When imu_in_body_frame=True, keep IMU in body frame to feed directly to body model
        if self.coordinate == "glob_coord" and not imu_in_body_frame:
            self.data["acc"] = self.data["gt_orientation"] @ self.data["acc"]
            self.data["gyro"] = self.data["gt_orientation"] @ self.data["gyro"]

        truncate_and_free(
            self.data,
            keys=["time", "acc", "gyro", "velocity", "gt_translation", "gt_orientation", "mask"],
            maximum_length=maximum_length
        )

    def get_length(self) -> int:
        return self.data["time"].shape[0]
    
if __name__ == "__main__":
    
    import time
    dataset = tlioCPF(
        data_root=os.path.join(os.environ.get("DATA_ROOT", "."), "tlio_dataset_cpf/test"),
        data_name="1024599341033662_cpf",
        coordinate="glob_coord",
        maximum_length=500000,
        remove_g=True
    )
    print("Dataset length:", dataset.get_length())
    

# python -m datasets.tliodataset_cpf