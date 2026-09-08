import argparse

import numpy as np
import torch
import torch.utils.data as Data
from pyhocon import ConfigFactory
from .dataset import Sequence, SeqeuncesDataset
import math
import pypose as pp

class SeqeuncesMotionDataset(SeqeuncesDataset):
    def __init__(
        self,
        data_set_config,
        mode=None,
        data_path=None,
        data_root=None,
        device="cuda:0",
    ):
        super().__init__(
        data_set_config=data_set_config,
        mode=mode,
        data_path=data_path,
        data_root=data_root,
        device=device,
        )
        print(f"******* Loading {data_set_config.mode} dataset *******")
        print(f"loaded: {data_set_config.data_list[0]['data_root']}")
        if "coordinate" in data_set_config:
            print(f"coordinate: {data_set_config.coordinate}")
        if "remove_g" in data_set_config and data_set_config.remove_g is True:
            print(f"gravity has been removed")
        if "rot_type" in data_set_config:
            if data_set_config.rot_type is None:
                print(f"using groundtruth orientation")
            elif data_set_config.rot_type.lower() == "airimu":
                print(f"Using AirIMU orientation loaded from {data_set_config.rot_path}.")
            elif data_set_config.rot_type.lower() == "integration":
                print(f"Using pre-integration orientation loaded from {data_set_config.rot_path}.")
        print(f"gravity: {data_set_config.gravity}")

    
    def load_data(self, seq, start_frame, end_frame):
        self.ts.append(seq.data["time"][start_frame:end_frame + 1])
        self.acc.append(seq.data["acc"][start_frame:end_frame])  # Same rate as pose
        self.gyro.append(seq.data["gyro"][start_frame:end_frame])  # Same rate as pose
        self.dt.append(seq.data["dt"][start_frame : end_frame + 1])
        self.gt_pos.append(seq.data["gt_translation"][start_frame : end_frame + 1])
        self.gt_ori.append(seq.data["gt_orientation"][start_frame : end_frame + 1])
        self.gt_velo.append(seq.data["velocity"][start_frame : end_frame + 1])

        # Handle optional baro data
        if "baro" in seq.data:
            self.baro.append(seq.data["baro"][start_frame : end_frame])
        else:
            # Create zero placeholder if baro data not available
            self.baro.append(torch.zeros(end_frame - start_frame, 1))

        # Handle optional mag data
        if "mag" in seq.data:
            self.mag.append(seq.data["mag"][start_frame : end_frame])
        else:
            # Create zero placeholder if mag data not available
            self.mag.append(torch.zeros(end_frame - start_frame, 1))

        # Handle optional left IMU data
        if "acc_left" in seq.data:
            self.acc_left.append(seq.data["acc_left"][start_frame : end_frame])
        else:
            self.acc_left.append(torch.zeros_like(seq.data["acc"][start_frame : end_frame]))

        if "gyro_left" in seq.data:
            self.gyro_left.append(seq.data["gyro_left"][start_frame : end_frame])
        else:
            self.gyro_left.append(torch.zeros_like(seq.data["gyro"][start_frame : end_frame]))

        # Load left IMU orientation if available
        if "orientation_left" in seq.data:
            if not hasattr(self, 'ori_left'):
                self.ori_left = []
            self.ori_left.append(seq.data["orientation_left"][start_frame : end_frame + 1])

        # Body joint labels (present only when the dataset config sets use_body: True)
        if "joint_pos" in seq.data and "joint_rot" in seq.data:
            if not hasattr(self, 'joint_pos'):
                self.joint_pos, self.joint_rot = [], []
            # sliced like gt velocity (T+1); network.get_label() trims it to the T output frames
            self.joint_pos.append(seq.data["joint_pos"][start_frame : end_frame + 1].cpu())
            self.joint_rot.append(seq.data["joint_rot"][start_frame : end_frame + 1].cpu())


    def construct_index_map(self, conf, data_root, data_name, seq_id):
        # Initialize start_frames tracking if not exists
        if not hasattr(self, 'start_frames'):
            self.start_frames = []
            
        # Pass mode and other parameters from self.conf, plus config parameters from conf if present
        extra_params = {}
        if 'mode' in self.conf:
            extra_params['mode'] = self.conf['mode']
        if 'coordinate' in self.conf:
            extra_params['coordinate'] = self.conf['coordinate']
        if 'gravity' in self.conf:
            extra_params['gravity'] = self.conf['gravity']
        if "remove_g" in self.conf:
            extra_params['remove_g'] = self.conf['remove_g']
        
        # Add window_size and step_size for tlio dataset
        if 'window_size' in conf:
            extra_params['window_size'] = conf['window_size']
        if 'step_size' in conf:
            extra_params['decimator'] = conf['step_size']  # step_size maps to decimator in tlio

        # Add dual-IMU parameters
        if 'load_left_imu' in self.conf:
            extra_params['load_left_imu'] = self.conf['load_left_imu']
        if 'load_left_imu_cpf' in self.conf:
            extra_params['load_left_imu_cpf'] = self.conf['load_left_imu_cpf']
        if 'load_left_imu_body' in self.conf:
            extra_params['load_left_imu_body'] = self.conf['load_left_imu_body']
        if 'rot_from_gyro' in self.conf:
            extra_params['rot_from_gyro'] = self.conf['rot_from_gyro']
        if 'imu_in_body_frame' in self.conf:
            extra_params['imu_in_body_frame'] = self.conf['imu_in_body_frame']
        if 'load_baro' in self.conf:
            extra_params['load_baro'] = self.conf['load_baro']
        if 'load_mag' in self.conf:
            extra_params['load_mag'] = self.conf['load_mag']
        if 'resample_hz' in self.conf:
            extra_params['resample_hz'] = self.conf['resample_hz']
        if 'use_body' in self.conf:
            extra_params['use_body'] = self.conf['use_body']


        seq = self.DataClass[conf.name](
            data_root, data_name, **extra_params
        )
        seq_len = seq.get_length() - 1
        window_size, step_size = conf.window_size, conf.step_size
        ## seting the starting and ending duration with different trianing mode
        start_frame, end_frame = 0, seq_len

        if self.mode == 'train_70':
            end_frame = np.floor(seq_len * 0.7).astype(int)
        elif self.mode == 'test_30':
            start_frame = np.floor(seq_len * 0.7).astype(int)

        _duration = end_frame - start_frame
        if self.mode == "inference":
            window_size = seq_len
            step_size = seq_len
            self.index_map = [[seq_id, 0, seq_len]]
        elif self.mode == "infevaluate":
            if _duration <= window_size:

                self.index_map += [[seq_id, 0, seq_len]]
            else:
                self.index_map += [
                    [seq_id, j, j + window_size]
                    for j in range(0, _duration - window_size, step_size)
                ]
            if self.index_map[-1][2] < _duration:
                self.index_map += [[seq_id, self.index_map[-1][2], seq_len]]

        elif self.mode == "evaluate":
            # adding the last piece for evaluation
            if _duration < window_size:
                print(f"Warning: Sequence {seq_id} has {_duration} frames but window_size is {window_size}. Using full sequence.")
                self.index_map += [[seq_id, 0, seq_len]]
            else:
                self.index_map += [
                    [seq_id, j, j + window_size]
                    for j in range(0, _duration - window_size, step_size)
                ]
        else:
            sub_index_map = [
                [seq_id, j, j + window_size]
                for j in range(0, _duration - window_size - step_size, step_size)
                if torch.all(seq.data["mask"][j : j + window_size])
            ]
            self.index_map += sub_index_map

        ## Loading the data from each sequence into
        self.load_data(seq, start_frame, end_frame)
        # Track the start_frame for this sequence
        self.start_frames.append(start_frame)
        

    def __getitem__(self, item):
        seq_id, abs_frame_id, abs_end_frame_id = self.index_map[item][0], self.index_map[item][1], self.index_map[item][2]
        
        # For relative indexing within the loaded data
        # We need to track the start_frame used during load_data
        if not hasattr(self, 'start_frames'):
            # If start_frames not tracked, assume data starts from 0
            pose_frame_id = abs_frame_id
            pose_end_frame_id = abs_end_frame_id
        else:
            # Adjust indices relative to the loaded data's start frame
            pose_frame_id = abs_frame_id - self.start_frames[seq_id]
            pose_end_frame_id = abs_end_frame_id - self.start_frames[seq_id]
        
        # Ensure pose indices are within bounds
        max_pose_len = len(self.gt_ori[seq_id])
        pose_frame_id = max(0, min(pose_frame_id, max_pose_len - 1))
        pose_end_frame_id = max(0, min(pose_end_frame_id, max_pose_len - 1))
        
        # Calculate actual window size
        pose_window_size = pose_end_frame_id - pose_frame_id
        
        # For IMU data: detect dual-rate and calculate proper indices
        max_imu_len = len(self.acc[seq_id])  
        max_pose_len = len(self.gt_ori[seq_id])
        

        imu_frame_id = pose_frame_id
        imu_end_frame_id = pose_end_frame_id
        imu_end_frame_id = max(0, min(imu_end_frame_id, max_imu_len))
        data = {
            'timestamp':  self.ts[seq_id][pose_frame_id:pose_end_frame_id + 1],
            'dt': self.dt[seq_id][pose_frame_id: pose_end_frame_id + 1],
            'acc': self.acc[seq_id][imu_frame_id:imu_end_frame_id],  # Windowed IMU data
            'gyro': self.gyro[seq_id][imu_frame_id:imu_end_frame_id],  # Windowed IMU data
            'rot': self.gt_ori[seq_id][pose_frame_id: pose_end_frame_id],  # Windowed pose data
            'baro': self.baro[seq_id][pose_frame_id: pose_end_frame_id],  
            'mag': self.mag[seq_id][pose_frame_id: pose_end_frame_id],  
            'acc_left': self.acc_left[seq_id][pose_frame_id: pose_end_frame_id],  
            'gyro_left': self.gyro_left[seq_id][pose_frame_id: pose_end_frame_id],  
        }

        # Add left IMU data if available
        if hasattr(self, 'acc_left') and len(self.acc_left) > seq_id:
            data['acc_left'] = self.acc_left[seq_id][imu_frame_id:imu_end_frame_id]
            data['gyro_left'] = self.gyro_left[seq_id][imu_frame_id:imu_end_frame_id]
        if hasattr(self, 'ori_left') and len(self.ori_left) > seq_id:
            data['rot_left'] = self.ori_left[seq_id][pose_frame_id: pose_end_frame_id]
        if hasattr(self, 'joint_pos') and len(self.joint_pos) > seq_id:
            data['joint_pos'] = self.joint_pos[seq_id][pose_frame_id:pose_end_frame_id + 1]
            data['joint_rot'] = self.joint_rot[seq_id][pose_frame_id:pose_end_frame_id + 1]


        # Ensure we have valid pose indices for initial state and labels
        pose_frame_id = max(0, min(pose_frame_id, len(self.gt_ori[seq_id]) - 1))
        pose_end_frame_id = max(0, min(pose_end_frame_id, len(self.gt_ori[seq_id]) - 1))

        init_state = {
            'init_rot':self.gt_ori[seq_id][pose_frame_id][None, ...],  # Contains gyro-integrated or SLAM rotation based on rot_from_gyro config
            'init_pos':self.gt_pos[seq_id][pose_frame_id][None, ...],
            'init_vel':self.gt_velo[seq_id][pose_frame_id][None, ...],
        }
        label = {
            'gt_pos':self.gt_pos[seq_id][pose_frame_id : pose_end_frame_id+1],
            'gt_rot':self.gt_ori[seq_id][pose_frame_id : pose_end_frame_id+1],  # GT labels: SLAM orientation
            'gt_vel':self.gt_velo[seq_id][pose_frame_id : pose_end_frame_id+1],
        }
        return {**data, **init_state, **label}

    def get_init_value(self):
        return {
            "pos": self.data["gt_translation"][:1],
            "rot": self.data["gt_orientation"][:1],
            "vel": self.data["velocity"][:1],
        }
if __name__ == "__main__":
    from datasets.dataset_utils import custom_collate

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/datasets/BaselineEuRoC.conf",
        help="config file path, i.e., configs/Euroc.conf",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="cuda or cpu")

    args = parser.parse_args()
    conf = ConfigFactory.parse_file(args.config)

    dataset = SeqeuncesMotionDataset(data_set_config=conf.train)
    loader = Data.DataLoader(
        dataset=dataset, batch_size=1, shuffle=False, collate_fn=custom_collate
    )

    for i, (data, init, _label) in enumerate(loader):
        for k in data:
            print(k, ":", data[k].shape)
        for k in init:
            print(k, ":", init[k].shape)
