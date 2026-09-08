import argparse
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.utils.data as Data
from pyhocon import ConfigFactory


class Sequence(ABC):
    # Dictionary to keep track of subclasses
    subclasses = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.subclasses[cls.__name__] = cls


class SeqDataset(Data.Dataset):
    def __init__(
        self,
        root,
        dataname,
        devive="cpu",
        name="ALTO",
        duration=200,
        step_size=200,
        mode="inference",
        drop_last=True,
        conf={},
    ):
        super().__init__()

        self.DataClass = Sequence.subclasses
        self.conf = conf
        self.seq = self.DataClass[name](root, dataname, **self.conf)
        self.data = self.seq.data
        self.seqlen = self.seq.get_length() - 1
        self.gravity = conf.gravity if "gravity" in conf.keys() else 9.81007
        self.interpolate = True

        if duration is None:
            self.duration = self.seqlen
        else:
            self.duration = duration

        if step_size is None:
            self.step_size = self.seqlen
        else:
            self.step_size = step_size

        self.data["acc_cov"] = 0.08 * torch.ones_like(self.data["acc"])
        self.data["gyro_cov"] = 0.006 * torch.ones_like(self.data["gyro"])

        start_frame = 0
        end_frame = self.seqlen

        if self.duration > (end_frame - start_frame):
            self.index_map = [[0, end_frame-start_frame]]
        else:
            self.index_map = [
                [i, i + self.duration]
                for i in range(0, end_frame - start_frame - self.duration, self.step_size)
            ]
        if (self.index_map[-1][-1] < end_frame) and (not drop_last):

            self.index_map.append([self.index_map[-1][-1], end_frame])

        self.index_map = np.array(self.index_map)

        loaded_param = f"loaded: {root}"
        if "calib" in self.conf:
            loaded_param += f", calib: {self.conf.calib}"
        loaded_param += f", interpolate: {self.interpolate}, gravity: {self.gravity}"
        print(loaded_param)

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, i):
        frame_id, end_frame_id = self.index_map[i]
        return {
            "timestamp": self.data['time'][frame_id+1: end_frame_id+1],
            "dt": self.data["dt"][frame_id:end_frame_id],
            "acc": self.data["acc"][frame_id:end_frame_id],
            "gyro": self.data["gyro"][frame_id:end_frame_id],
            "rot": self.data["gt_orientation"][frame_id:end_frame_id],
            "gt_pos": self.data["gt_translation"][frame_id + 1 : end_frame_id + 1],
            "gt_rot": self.data["gt_orientation"][frame_id + 1 : end_frame_id + 1],
            "gt_vel": self.data["velocity"][frame_id + 1 : end_frame_id + 1],
            "init_pos": self.data["gt_translation"][frame_id][None, ...],
            "init_rot": self.data["gt_orientation"][frame_id:end_frame_id],
            "init_vel": self.data["velocity"][frame_id][None, ...],
        }

    def get_init_value(self):
        return {
            "pos": self.data["gt_translation"][:1],
            "rot": self.data["gt_orientation"][:1],
            "vel": self.data["velocity"][:1],
        }

    def get_mask(self):
        return self.data["mask"]

    def get_gravity(self):
        return self.gravity


class SeqInfDataset(SeqDataset):
    def __init__(
        self,
        root,
        dataname,
        inference_state,
        device="cpu",
        name="ALTO",
        duration=200,
        step_size=200,
        drop_last=True,
        mode="inference",
        usecov=True,
        useraw=False,
        usetimecut=False,
        conf={},
    ):
        super().__init__(
            root, dataname, device, name, duration, step_size, mode, drop_last, conf
        )
        time_cut = 0
        if usetimecut:
            time_cut = self.seq.time_cut
        if "correction_acc" in inference_state.keys() and not useraw:
            self.data["acc"][:-1] += inference_state["correction_acc"][:, time_cut:].cpu()[0]
            self.data["gyro"][:-1] += inference_state["correction_gyro"][:, time_cut:].cpu()[0]

        if "gyro_bias" in inference_state.keys():
            print("adapted gyro bias: ", inference_state["gyro_bias"][time_cut:].cpu())
            self.data["gyro"][:-1] -= inference_state["gyro_bias"][time_cut:].cpu()
        if "acc_bias" in inference_state.keys():
            print("adapted acc bias: ", inference_state["acc_bias"][time_cut:].cpu())
            self.data["acc"][:-1] -= inference_state["acc_bias"][time_cut:].cpu()

        if "adapt_acc" in inference_state.keys():
            self.data["acc"][:-1] -= np.array(inference_state["adapt_acc"][time_cut:])
            self.data["gyro"][:-1] -= np.array(inference_state["adapt_gyro"][time_cut:])

        if "acc_cov" in inference_state.keys() and usecov:
            self.data["acc_cov"] = inference_state["acc_cov"][0][time_cut:]

        if "gyro_cov" in inference_state.keys() and usecov:
            self.data["gyro_cov"] = inference_state["gyro_cov"][0][time_cut:]

    def get_bias(self):
        return {"gyro_bias": self.data["g_b"][:-1], "acc_bias": self.data["a_b"][:-1]}
    
    
    def __getitem__(self, i):
        frame_id, end_frame_id = self.index_map[i]
        return {
            "acc_cov": self.data["acc_cov"][frame_id:end_frame_id] if "acc_cov" in self.data.keys() else None,
            "gyro_cov": self.data["gyro_cov"][frame_id:end_frame_id] if "gyro_cov" in self.data.keys() else None,
            "timestamp": self.data['time'][frame_id+1: end_frame_id+1],
            "dt": self.data["dt"][frame_id:end_frame_id],
            "acc": self.data["acc"][frame_id:end_frame_id],
            "gyro": self.data["gyro"][frame_id:end_frame_id],
            "rot": self.data["gt_orientation"][frame_id:end_frame_id],
            "gt_pos": self.data["gt_translation"][frame_id + 1 : end_frame_id + 1],
            "gt_rot": self.data["gt_orientation"][frame_id + 1 : end_frame_id + 1],
            "gt_vel": self.data["velocity"][frame_id + 1 : end_frame_id + 1],
            "init_pos": self.data["gt_translation"][frame_id][None, ...],
            "init_rot": self.data["gt_orientation"][frame_id:end_frame_id],
            "init_vel": self.data["velocity"][frame_id][None, ...],
        }


class SeqeuncesDataset(Data.Dataset):
    """
    For the purpose of training and inferering
    1. Abandon the features of the last time frame, since there are no ground truth pose and dt
     to integrate the imu data of the last frame. So the length of the dataset is seq.get_length() - 1
    """

    def __init__(
        self,
        data_set_config,
        mode=None,
        data_path=None,
        data_root=None,
        device="cuda:0",
    ):
        super(SeqeuncesDataset, self).__init__()
        (
            self.ts,
            self.dt,
            self.acc,
            self.gyro,
            self.gt_pos,
            self.gt_ori,
            self.gt_velo,
            self.baro,
            self.acc_left,
            self.gyro_left,
            self.mag,
            self.index_map,
            self.seq_idx,
        ) = ([], [], [], [], [], [], [], [] , [], [],[],[], 0)
        self.uni = torch.distributions.uniform.Uniform(-torch.ones(1), torch.ones(1))
        self.device = device
        self.interpolate = True
        self.conf = data_set_config
        self.gravity = self.conf.gravity if "gravity" in self.conf.keys() else 9.81007
        self.weights = []

        if mode is None:
            self.mode = data_set_config.mode
        else:
            self.mode = mode

        self.DataClass = Sequence.subclasses
        ## the design of datapath provide a quick way to revisit a specific sequence, but introduce some inconsistency
        if data_path is None:
            for conf in data_set_config.data_list:
                if isinstance(conf.data_drive, list):
                    if len(conf.data_drive) == 0:
                        if conf.name == "nymeria" or conf.name == "aria" or conf.name == "ariaCPF" or conf.name == "nymeria_rwrist":
                            # Empty list means load all pickle files from directory
                            import glob
                            import os
                            pickle_files = sorted(glob.glob(os.path.join(conf["data_root"], "*.pkl")))
                            if not pickle_files:
                                pickle_files = sorted(glob.glob(os.path.join(conf["data_root"], "*.pickle")))
                            print(f"Loading all {len(pickle_files)} sequences from {conf['data_root']}")
                            from tqdm.auto import tqdm
                            for pkl_path in tqdm(pickle_files, desc="Indexing pickle files", unit="file", total=len(pickle_files)):
                                    # Extract just the filename without extension
                                path = os.path.splitext(os.path.basename(pkl_path))[0]
                                self.construct_index_map(conf, conf["data_root"], path, self.seq_idx)
                                self.seq_idx += 1
                        elif conf.name == "tlioCPF":
                            import glob
                            import os
                            # Look for .npz files instead of directories
                            npz_files = sorted(glob.glob(os.path.join(conf["data_root"], "*.npz")))
                            from tqdm.auto import tqdm
                            for npz_path in tqdm(npz_files, desc="Indexing npz files", unit="file", total=len(npz_files)):
                                # Extract filename without extension
                                data_name = os.path.splitext(os.path.basename(npz_path))[0]
                                # # Remove _cpf suffix if present
                                # if data_name.endswith('_cpf'):
                                #     data_name = data_name[:-4]
                                self.construct_index_map(conf, conf["data_root"], data_name, self.seq_idx)
                                self.seq_idx += 1
                        elif conf.name == "inrtl":
                            import glob
                            import os
                            from tqdm.auto import tqdm
                            npz_files = sorted(glob.glob(os.path.join(conf["data_root"], "*_cpf.npz")))
                            print(f"Loading {len(npz_files)} INRTL sequences from {conf['data_root']}")
                            for npz_path in tqdm(npz_files, desc="Indexing INRTL sequences", unit="file"):
                                data_name = os.path.splitext(os.path.basename(npz_path))[0]
                                if data_name.endswith("_cpf"):
                                    data_name = data_name[:-4]
                                self.construct_index_map(conf, conf["data_root"], data_name, self.seq_idx)
                                self.seq_idx += 1
                        else:
                            raise ValueError("wrong dataset name")
                    else:
                        # Use explicit list
                        for path in conf.data_drive:
                            self.construct_index_map(
                                conf, conf["data_root"], path, self.seq_idx
                            )
                            self.seq_idx += 1
                elif isinstance(conf.data_drive, str) and conf.data_drive.lower().endswith('.txt'):
                    with open(conf.data_drive, 'r') as f:
                        for p in f:
                            path = p.strip() 
                            self.construct_index_map(
                                conf, conf["data_root"], path, self.seq_idx
                            )
                            self.seq_idx += 1
                else:
                    raise ValueError("Wrong conf.data_drive")
                    
        ## the design of dataroot provide a quick way to introduce multiple sequences in eval set, but introduce some inconsistency
        elif data_root is None:
            conf = data_set_config.data_list[0]
            self.construct_index_map(conf, conf["data_root"], data_path, self.seq_idx)
            self.seq_idx += 1
        else:
            conf = data_set_config.data_list[0]
            self.construct_index_map(conf, data_root, data_path, self.seq_idx)
            self.seq_idx += 1


    def load_data(self, seq, start_frame, end_frame):
        if "time" in seq.data.keys():
            self.ts.append(seq.data["time"][start_frame:end_frame + 1])
        self.acc.append(seq.data["acc"][start_frame:end_frame])
        self.gyro.append(seq.data["gyro"][start_frame:end_frame])
        # the groud truth state should include the init state and integrated state, thus has one
        # frame than imu data
        self.dt.append(seq.data["dt"][start_frame : end_frame + 1])
        self.gt_pos.append(seq.data["gt_translation"][start_frame : end_frame + 1])
        self.gt_ori.append(seq.data["gt_orientation"][start_frame : end_frame + 1])
        self.gt_velo.append(seq.data["velocity"][start_frame : end_frame + 1])

    def construct_index_map(self, conf, data_root, data_name, seq_id):
        # Merge global config with item-specific config, giving priority to item-specific
        merged_conf = dict(self.conf)
        merged_conf.update(dict(conf))
        seq = self.DataClass[conf.name](
            data_root, data_name, interpolate=self.interpolate, **merged_conf
        )
        seq_len = seq.get_length() - 1  # abandon the last imu features
        window_size, step_size = conf.window_size, conf.step_size

        # Skip sequences that are smaller than window_size
        if seq_len < window_size:
            print(f"Skipping sequence {data_name}: length {seq_len} < window_size {window_size}")
            return

        ## seting the starting and ending duration with different trianing mode
        start_frame, end_frame = 0, seq_len
        
        # Check for start_offset parameter
        print(f"DEBUG: Checking for start_offset in conf. Conf keys: {list(conf.keys()) if hasattr(conf, 'keys') else 'No keys method'}")
        if "start_offset" in conf:
            start_frame = max(0, min(conf.start_offset, seq_len - 1))
            print(f"Applied start_offset: {start_frame} frames")
        else:
            print(f"DEBUG: start_offset not found in conf")

        if self.mode == "train_half":
            end_frame = np.floor(seq_len * 0.5).astype(int)
        elif self.mode == "test_half":
            start_frame = np.floor(seq_len * 0.5).astype(int)
        elif self.mode == "train_1m":
            end_frame = 12000
        elif self.mode == "test_1m":
            start_frame = 12000
        elif self.mode == "mini":  # For the purpse of debug
            end_frame = 1000

        _duration = end_frame - start_frame
        if self.mode == "inference":
            window_size = seq_len
            step_size = seq_len
            self.index_map = [[seq_id, 0, seq_len]]
        elif self.mode == "infevaluate":
            self.index_map += [
                [seq_id, start_frame + j, start_frame + j + window_size]
                for j in range(0, _duration - window_size, step_size)
            ]
            if self.index_map[-1][2] < start_frame + _duration:
                print(self.index_map[-1][2])
                self.index_map += [[seq_id, self.index_map[-1][2], start_frame + _duration]]
        elif self.mode == "evaluate":
            # adding the last piece for evaluation
            self.index_map += [
                [seq_id, j, j + window_size]
                for j in range(0, _duration - window_size, step_size)
            ]
        elif self.mode == "train_half_random":
            np.random.seed(1)
            window_group_size = 3000
            selected_indices = [
                j for j in range(0, _duration - window_group_size, window_group_size)
            ]
            np.random.shuffle(selected_indices)
            indices_num = len(selected_indices)
            for w in selected_indices[: np.floor(indices_num * 0.5).astype(int)]:
                self.index_map += [
                    [seq_id, j, j + window_size]
                    for j in range(w, w + window_group_size - window_size, step_size)
                ]
        elif self.mode == "test_half_random":
            np.random.seed(1)
            window_group_size = 3000
            selected_indices = [
                j for j in range(0, _duration - window_group_size, window_group_size)
            ]
            np.random.shuffle(selected_indices)
            indices_num = len(selected_indices)
            for w in selected_indices[np.floor(indices_num * 0.5).astype(int) :]:
                self.index_map += [
                    [seq_id, j, j + window_size]
                    for j in range(w, w + window_group_size - window_size, step_size)
                ]
        else:
            ## applied the mask if we need the training.
            self.index_map += [
                [seq_id, j, j + window_size]
                for j in range(0, _duration - window_size, step_size)
                if torch.all(seq.data["mask"][j : j + window_size])
            ]

        ## Loading the data from each sequence into
        self.load_data(seq, start_frame, end_frame)

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, item):
        seq_id, frame_id, end_frame_id = (
            self.index_map[item][0],
            self.index_map[item][1],
            self.index_map[item][2],
        )
        data = {
            "dt": self.dt[seq_id][frame_id:end_frame_id],
            "acc": self.acc[seq_id][frame_id:end_frame_id],
            "gyro": self.gyro[seq_id][frame_id:end_frame_id],
            "rot": self.gt_ori[seq_id][frame_id:end_frame_id],
        }
        init_state = {
            "init_rot": self.gt_ori[seq_id][frame_id][None, ...],
            "init_pos": self.gt_pos[seq_id][frame_id][None, ...],
            "init_vel": self.gt_velo[seq_id][frame_id][None, ...],
        }
        label = {
            "gt_pos": self.gt_pos[seq_id][frame_id + 1 : end_frame_id + 1],
            "gt_rot": self.gt_ori[seq_id][frame_id + 1 : end_frame_id + 1],
            "gt_vel": self.gt_velo[seq_id][frame_id + 1 : end_frame_id + 1],
        }

        return {**data, **init_state, **label}

    def get_dtype(self):
        return self.acc[0].dtype

    def get_gravity(self):
        return self.gravity

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
    print(args)
    conf = ConfigFactory.parse_file(args.config)

    dataset = SeqeuncesDataset(data_set_config=conf.train)
    loader = Data.DataLoader(
        dataset=dataset, batch_size=1, shuffle=False, collate_fn=custom_collate
    )

    for i, (data, init, _label) in enumerate(loader):
        for k in data:
            print(k, ":", data[k].shape)
        for k in init:
            print(k, ":", init[k].shape)
