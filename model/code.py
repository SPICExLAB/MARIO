import numpy as np
import pypose as pp
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CNNEncoder(nn.Module):
    def __init__(self, duration = 1, k_list = [7, 7, 7, 7], c_list = [6, 16, 32, 64, 128], 
                        s_list = [1, 1, 1, 1], p_list = [3, 3, 3, 3]):
        super(CNNEncoder, self).__init__()
        self.duration = duration
        self.k_list, self.c_list, self.s_list, self.p_list = k_list, c_list, s_list, p_list
        layers = []

        for i in range(len(self.c_list) - 1):
            layers.append(torch.nn.Conv1d(self.c_list[i], self.c_list[i+1], self.k_list[i], \
                stride=self.s_list[i], padding=self.p_list[i]))
            layers.append(torch.nn.BatchNorm1d(self.c_list[i+1]))
            layers.append(torch.nn.GELU())
        layers.append(torch.nn.Dropout(0.5))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class CNNEncoder_keep_len(nn.Module):
    def __init__(
        self,
        duration=1,
        k_list=[7, 7, 7, 7],
        c_list=[6, 16, 32, 64, 128],
        s_list=None,                  # if None and preserve_len=True -> all 1s
        p_list=None,                  # if None and preserve_len=True -> computed "same"
        preserve_len=False            # <— set True to keep T unchanged
    ):
        super().__init__()
        self.duration = duration
        self.k_list = k_list
        self.c_list = c_list
        self.preserve_len = preserve_len

        if s_list is None:
            s_list = [1] * (len(c_list) - 1) if preserve_len else [1] * (len(c_list) - 1)
        if p_list is None:
            if preserve_len:
                # "same" padding for odd kernels (works with your 7s)
                p_list = [(k - 1) // 2 for k in k_list]
            else:
                p_list = [0] * (len(c_list) - 1)

        self.s_list = s_list
        self.p_list = p_list

        layers = []
        for i in range(len(self.c_list) - 1):
            k = self.k_list[i]
            s = self.s_list[i]
            p = self.p_list[i] if not isinstance(self.p_list[i], str) else self.p_list[i]

            layers += [
                nn.Conv1d(self.c_list[i], self.c_list[i+1], kernel_size=k, stride=s, padding=p),
                nn.BatchNorm1d(self.c_list[i+1]),
                nn.GELU()
            ]
        layers.append(nn.Dropout(0.5))
        self.net = nn.Sequential(*layers)

    def forward(self, x):   # x: [B, C_in, T]
        return self.net(x)  # -> [B, C_out, T] if preserve_len=True and odd kernels

class CodeNetMotion(torch.nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.conf = conf    
        self.k_list = [7, 7]
        self.p_list = [3, 3]
        self.s_list = [3, 3]
        self.cnn = CNNEncoder(c_list=[6, 32, 64], k_list=self.k_list, s_list=self.s_list, p_list=self.p_list)# (N,F/8,64)
        self.gru1 = nn.GRU(input_size = 64, hidden_size =64, num_layers = 1, batch_first = True,bidirectional=True)
        self.gru2 = nn.GRU(input_size = 128, hidden_size =128, num_layers = 1, batch_first = True,bidirectional=True)
        self.veldecoder = nn.Sequential(nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 3))
        self.velcov_decoder = nn.Sequential(nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 3))

    def encoder(self, x):
        x = self.cnn(x.transpose(-1, -2)).transpose(-1, -2)
        x, _ = self.gru1(x)
        x, _ = self.gru2(x)
        return x

    def cov_decoder(self, x):
        cov = torch.exp(self.velcov_decoder(x) - 5.0)
        return cov

    def decoder(self, x):
        vel = self.veldecoder(x)
        return vel
    
    def get_label(self, gt_label):   
        s_idx = (self.k_list[0] - self.p_list[0]) + self.s_list[0] * (self.k_list[1] - 1 -self.p_list[1]) + 1
        select_label = gt_label[:, s_idx::self.s_list[0]* self.s_list[1],:]
        L_out = (gt_label.shape[1] -1 - 1) // self.s_list[0] // self.s_list[1] + 1
        diff = L_out - select_label.shape[1]
        if diff > 0:
            select_label = torch.cat((select_label,gt_label[:,-1:,:].repeat(1,diff , 1)),dim = 1)
        return select_label
    
    def forward(self, data, rot=None):
        feature = torch.cat([data["acc"], data["gyro"]], dim = -1)
        feature = self.encoder(feature)
        net_vel = self.decoder(feature)
   
        cov = None
        if self.conf.propcov:
            cov = self.cov_decoder(feature)
        return {"cov": cov, 'net_vel': net_vel}

class CodeNetMotionwithRot(CodeNetMotion):
    def __init__(self, conf):
        super().__init__(conf)
        self.conf = conf
        self.interval = 9
        self.k_list = [7, 7]
        self.padding_num = 3
        self.s_list = [3, 3]

        self.feature_encoder = CNNEncoder(c_list=[6, 32, 64], k_list=self.k_list, s_list=self.s_list, p_list=[3,3])# (N,F/8,64)
        self.ori_encoder = CNNEncoder(c_list=[3, 32, 64], k_list=self.k_list, s_list=self.s_list, p_list=[3,3])# (N,F/8,64)
        self.gru1 = nn.GRU(input_size = 64, hidden_size =64, num_layers = 1, batch_first = True,bidirectional=True)
        self.gru2 = nn.GRU(input_size = 128, hidden_size =128, num_layers = 1, batch_first = True,bidirectional=True)
        self.fcn1 = nn.Sequential(nn.Linear(128, 128))
        self.batchnorm1 = torch.nn.BatchNorm1d(128)
        self.fcn2 = nn.Sequential(nn.Linear(128, 64))
        self.batchnorm2 = torch.nn.BatchNorm1d(64)

        self.gelu = nn.GELU()
        self.veldecoder = nn.Sequential(nn.Linear(256, 128),nn.GELU(), nn.Linear(128, 3))
        self.velcov_decoder = nn.Sequential(nn.Linear(256, 128),nn.GELU(), nn.Linear(128, 3))

    def encoder(self, feature, ori, return_features=False):
        x1 = self.feature_encoder(feature.transpose(-1, -2)).transpose(-1, -2)
        x2 = self.ori_encoder(ori.transpose(-1,-2)).transpose(-1, -2)
        x_cat = torch.cat([x1, x2], dim = -1)

        x = self.fcn2(x_cat)
        x = self.batchnorm2(x.transpose(-1,-2)).transpose(-1,-2)
        x = self.gelu(x)

        # Save intermediate features
        cnn_features = x_cat[:, :, :64]  # First 64 channels from CNN concat

        x, _ = self.gru1(x)
        gru1_features = x  # [B, T', 128]

        x, _ = self.gru2(x)
        gru2_features = x  # [B, T', 256]

        if return_features:
            return {
                'cnn_features': cnn_features,
                'gru1_features': gru1_features,
                'gru2_features': gru2_features,
                'final_features': gru2_features
            }
        return gru2_features

    def forward(self, data, rot=None, return_features=False):
        assert rot is not None
        feature = torch.cat([data["acc"], data["gyro"]], dim = -1)
        enc = self.encoder(feature, rot, return_features=return_features)

        if return_features:
            net_vel = self.decoder(enc['final_features'])
            cov = None
            if self.conf.propcov:
                cov = self.cov_decoder(enc['final_features'])
            return {
                "cov": cov,
                'net_vel': net_vel,
                'cnn_features': enc['cnn_features'],
                'gru1_features': enc['gru1_features'],
                'gru2_features': enc['gru2_features'],
                'lstm_features': enc['gru2_features']  # Alias for wrapper compatibility
            }
        else:
            net_vel = self.decoder(enc)
            cov = None
            if self.conf.propcov:
                cov = self.cov_decoder(enc)
            return {"cov": cov, 'net_vel': net_vel}

class CodeNetMotionwithRot_Pose(CodeNetMotion):
    def __init__(self, conf):
        super().__init__(conf)
        self.conf = conf
        self.interval = 9
        self.k_list = [7, 7]
        self.padding_num = 3
        self.s_list = [3, 3]

        self.feature_encoder = CNNEncoder(c_list=[6, 32, 64], k_list=self.k_list, s_list=self.s_list, p_list=[3,3])# (N,F/8,64)
        self.ori_encoder = CNNEncoder(c_list=[3, 32, 64], k_list=self.k_list, s_list=self.s_list, p_list=[3,3])# (N,F/8,64)
        self.joint_encoder = CNNEncoder(c_list=[6*9, 64, 64], k_list=self.k_list, s_list=self.s_list, p_list=[3,3])# (N,F/8,64)
        self.gru1 = nn.GRU(input_size = 64, hidden_size =64, num_layers = 1, batch_first = True,bidirectional=True)
        self.gru2 = nn.GRU(input_size = 128, hidden_size =128, num_layers = 1, batch_first = True,bidirectional=True)
        self.fcn1 = nn.Sequential(nn.Linear(128, 128))
        self.batchnorm1 = torch.nn.BatchNorm1d(128)
        self.fcn2 = nn.Sequential(nn.Linear(128+64, 64))
        self.batchnorm2 = torch.nn.BatchNorm1d(64)

        self.gelu = nn.GELU()
        self.veldecoder = nn.Sequential(nn.Linear(256, 128),nn.GELU(), nn.Linear(128, 3))
        self.velcov_decoder = nn.Sequential(nn.Linear(256, 128),nn.GELU(), nn.Linear(128, 3))

    def keep_legs_only(self, joint_rot, include_pelvis=False):
        # joint_rot: [B, K, 24*6]
        B, K, _ = joint_rot.shape
        jr = joint_rot.view(B, K, 24, 6)
        leg_idx = [1,4,7,10,  2,5,8,11]  # L hip/knee/ankle/foot, R hip/knee/ankle/foot
        if include_pelvis:
            leg_idx = [0] + leg_idx
        keep = torch.tensor(leg_idx, device=joint_rot.device, dtype=torch.long)

        jr_kept = torch.index_select(jr, dim=2, index=keep).reshape(B, K, keep.numel()*6)
        return jr_kept, keep

    def encoder(self, feature, ori, joint_rot, return_features=False):
        x1 = self.feature_encoder(feature.transpose(-1, -2)).transpose(-1, -2)
        x2 = self.ori_encoder(ori.transpose(-1,-2)).transpose(-1, -2)
        joint_rot, kept_idx = self.keep_legs_only(joint_rot, include_pelvis=True)
        x3 = self.joint_encoder(joint_rot.transpose(-1,-2)).transpose(-1, -2)
        x_cat = torch.cat([x1, x2, x3], dim = -1)

        x = self.fcn2(x_cat)
        x = self.batchnorm2(x.transpose(-1,-2)).transpose(-1,-2)
        x = self.gelu(x)

        # Save intermediate features
        cnn_features = x_cat[:, :, :64]  # First 64 channels from CNN concat

        x, _ = self.gru1(x)
        gru1_features = x  # [B, T', 128]

        x, _ = self.gru2(x)
        gru2_features = x  # [B, T', 256]

        if return_features:
            return {
                'cnn_features': cnn_features,
                'gru1_features': gru1_features,
                'gru2_features': gru2_features,
                'final_features': gru2_features
            }
        return gru2_features

    def forward(self, data, rot=None, joint_rot=None, return_features=False):
        assert rot is not None
        feature = torch.cat([data["acc"], data["gyro"]], dim = -1)
        enc = self.encoder(feature, rot, joint_rot=joint_rot, return_features=return_features)

        # if return_features:
        #     net_vel = self.decoder(enc['final_features'])
        #     cov = None
        #     if self.conf.propcov:
        #         cov = self.cov_decoder(enc['final_features'])
        #     return {
        #         "cov": cov,
        #         'net_vel': net_vel,
        #         'cnn_features': enc['cnn_features'],
        #         'gru1_features': enc['gru1_features'],
        #         'gru2_features': enc['gru2_features'],
        #         'lstm_features': enc['gru2_features']  # Alias for wrapper compatibility
        #     }
        # else:
        #     net_vel = self.decoder(enc)
        #     cov = None
        #     if self.conf.propcov:
        #         cov = self.cov_decoder(enc)
        #     return {"cov": cov, 'net_vel': net_vel}
        net_vel = self.decoder(enc)
        cov = None
        if self.conf.propcov:
            cov = self.cov_decoder(enc)
        return {"cov": cov, 'net_vel': net_vel}

class CodeNetMotionwithPose_v4(CodeNetMotion):
    def __init__(self, conf):
        super().__init__(conf)
        self.conf = conf    
        self.interval = 9
        self.k_list = [7, 7]
        self.padding_num = 3
        self.s_list = [3, 3]
        
        self.feature_encoder = CNNEncoder_keep_len(
                                    c_list=[6, 32, 64],
                                    k_list=self.k_list,        
                                    preserve_len=True        
                                )
        self.ori_encoder = CNNEncoder_keep_len(
                                    c_list=[3, 32, 64],
                                    k_list=self.k_list,
                                    preserve_len=True           
                                )
        self.gru1 = nn.GRU(input_size = 64, hidden_size =64, num_layers = 1, batch_first = True,bidirectional=True)
        self.gru2 = nn.GRU(input_size = 128, hidden_size =128, num_layers = 1, batch_first = True,bidirectional=True)
        self.fcn1 = nn.Sequential(nn.Linear(128, 128))
        self.batchnorm1 = torch.nn.BatchNorm1d(128)
        self.fcn2 = nn.Sequential(nn.Linear(128, 64))
        self.batchnorm2 = torch.nn.BatchNorm1d(64)
        
        self.gelu = nn.GELU()

        self.veldecoder = nn.Sequential(nn.Linear(256, 128),nn.GELU(), nn.Linear(128, 3))
        self.velcov_decoder = nn.Sequential(nn.Linear(256, 128),nn.GELU(), nn.Linear(128, 3))
        
        self.joint_pos_gru = nn.GRU(input_size=256, hidden_size=128, num_layers=1, batch_first=True)
        self.joint_pos_proj = nn.Linear(128, 24 * 3)

        # Joint rotation decoder: GRU + Linear → [B, N, 24*6]
        self.joint_rot_gru = nn.GRU(input_size=24 * 3, hidden_size=128, num_layers=1, batch_first=True)
        self.joint_rot_proj = nn.Linear(128, 24 * 6)
        

    def encoder(self, feature, ori):
        x1 = self.feature_encoder(feature.transpose(-1, -2)).transpose(-1, -2)
        x2 = self.ori_encoder(ori.transpose(-1,-2)).transpose(-1, -2) 
        x = torch.cat([x1, x2], dim = -1)
        
        x = self.fcn2(x)
        x = self.batchnorm2(x.transpose(-1,-2)).transpose(-1,-2)
        x = self.gelu(x)
        x, _ = self.gru1(x)
        x, _ = self.gru2(x)
        return x

    def get_label(self, gt_label):
        """Adjust label to match network output dimension"""
        # Since we use rot[:,:-1,:] which removes last timestamp,
        # we need to match velocity labels accordingly
        return gt_label[:, :-1, :]  # Remove last velocity to match
    
    def forward(self, data, rot=None):
        assert rot is not None
        feature = torch.cat([data["acc"], data["gyro"]], dim = -1)
        feature = self.encoder(feature, rot)
        net_vel = self.decoder(feature)

        # Joint position prediction
        joint_pos_feat, _ = self.joint_pos_gru(feature)           # [B, N, 128]
        joint_pos = self.joint_pos_proj(joint_pos_feat)           # [B, N, 24*3]

        # Joint rotation prediction         
        joint_rot_feat, _ = self.joint_rot_gru(joint_pos)   # [B, N, 128]
        joint_rot = self.joint_rot_proj(joint_rot_feat)           # [B, N, 24*6]
   
        #covariance propagation
        cov = None
        if self.conf.propcov:
            cov = self.cov_decoder(feature)
        return {"cov": cov, 
                'net_vel': net_vel,
                "joint_pos": joint_pos,
                "joint_rot": joint_rot}

class CodeNetMotionwithRot_Baro(CodeNetMotion):
    def __init__(self, conf):
        super().__init__(conf)
        self.conf = conf    
        self.interval = 9
        self.k_list = [7, 7]
        self.padding_num = 3
        self.s_list = [3, 3]
        
        self.feature_encoder = CNNEncoder(c_list=[6, 32, 64], k_list=self.k_list, s_list=self.s_list, p_list=[3,3])# (N,F/8,64)
        self.ori_encoder = CNNEncoder(c_list=[3, 32, 64], k_list=self.k_list, s_list=self.s_list, p_list=[3,3])# (N,F/8,64)
        self.baro_encoder = CNNEncoder(c_list=[1, 32, 64], k_list=self.k_list, s_list=self.s_list, p_list=[3,3])# (N,F/8,64)
        self.gru1 = nn.GRU(input_size = 64, hidden_size =64, num_layers = 1, batch_first = True,bidirectional=True)
        self.gru2 = nn.GRU(input_size = 128, hidden_size =128, num_layers = 1, batch_first = True,bidirectional=True)
        self.fcn1 = nn.Sequential(nn.Linear(128, 128))
        self.batchnorm1 = torch.nn.BatchNorm1d(128)
        self.fcn2 = nn.Sequential(nn.Linear(128+64, 64))
        self.batchnorm2 = torch.nn.BatchNorm1d(64)
        
        self.gelu = nn.GELU()
        self.veldecoder = nn.Sequential(nn.Linear(256, 128),nn.GELU(), nn.Linear(128, 3))
        self.velcov_decoder = nn.Sequential(nn.Linear(256, 128),nn.GELU(), nn.Linear(128, 3))

    def encoder(self, feature, ori, baro):
        x1 = self.feature_encoder(feature.transpose(-1, -2)).transpose(-1, -2)
        x2 = self.ori_encoder(ori.transpose(-1,-2)).transpose(-1, -2) 
        x3 = self.baro_encoder(baro.transpose(-1,-2)).transpose(-1, -2)
        x = torch.cat([x1, x2, x3], dim = -1)
        
        x = self.fcn2(x)
        x = self.batchnorm2(x.transpose(-1,-2)).transpose(-1,-2)
        x = self.gelu(x)
        x, _ = self.gru1(x)
        x, _ = self.gru2(x)
        return x
    
    def forward(self, data, rot=None):
        assert rot is not None
        feature = torch.cat([data["acc"], data["gyro"]], dim = -1)
        feature = self.encoder(feature, rot, data["baro"])
        net_vel = self.decoder(feature)
   
        #covariance propagation
        cov = None
        if self.conf.propcov:
            cov = self.cov_decoder(feature)
        return {"cov": cov, 'net_vel': net_vel}

class CodeNetMotionwithRot_Mag(CodeNetMotion):
    def __init__(self, conf):
        super().__init__(conf)
        self.conf = conf    
        self.interval = 9
        self.k_list = [7, 7]
        self.padding_num = 3
        self.s_list = [3, 3]
        
        self.feature_encoder = CNNEncoder(c_list=[6, 32, 64], k_list=self.k_list, s_list=self.s_list, p_list=[3,3])# (N,F/8,64)
        self.ori_encoder = CNNEncoder(c_list=[3, 32, 64], k_list=self.k_list, s_list=self.s_list, p_list=[3,3])# (N,F/8,64)
        self.mag_encoder = CNNEncoder(c_list=[1, 32, 64], k_list=self.k_list, s_list=self.s_list, p_list=[3,3])# (N,F/8,64)
        self.gru1 = nn.GRU(input_size = 64, hidden_size =64, num_layers = 1, batch_first = True,bidirectional=True)
        self.gru2 = nn.GRU(input_size = 128, hidden_size =128, num_layers = 1, batch_first = True,bidirectional=True)
        self.fcn1 = nn.Sequential(nn.Linear(128, 128))
        self.batchnorm1 = torch.nn.BatchNorm1d(128)
        self.fcn2 = nn.Sequential(nn.Linear(128+64, 64))
        self.batchnorm2 = torch.nn.BatchNorm1d(64)
        
        self.gelu = nn.GELU()
        self.veldecoder = nn.Sequential(nn.Linear(256, 128),nn.GELU(), nn.Linear(128, 3))
        self.velcov_decoder = nn.Sequential(nn.Linear(256, 128),nn.GELU(), nn.Linear(128, 3))

    def encoder(self, feature, ori, mag):
        x1 = self.feature_encoder(feature.transpose(-1, -2)).transpose(-1, -2)
        x2 = self.ori_encoder(ori.transpose(-1,-2)).transpose(-1, -2) 
        x3 = self.mag_encoder(mag.transpose(-1,-2)).transpose(-1, -2)
        x = torch.cat([x1, x2, x3], dim = -1)
        
        x = self.fcn2(x)
        x = self.batchnorm2(x.transpose(-1,-2)).transpose(-1,-2)
        x = self.gelu(x)
        x, _ = self.gru1(x)
        x, _ = self.gru2(x)
        return x
    
    def forward(self, data, rot=None):
        assert rot is not None
        feature = torch.cat([data["acc"], data["gyro"]], dim = -1)
        feature = self.encoder(feature, rot, data["mag"])
        net_vel = self.decoder(feature)
   
        #covariance propagation
        cov = None
        if self.conf.propcov:
            cov = self.cov_decoder(feature)
        return {"cov": cov, 'net_vel': net_vel}

class CodeNetMotionwithRot_LeftIMU(CodeNetMotion):
    def __init__(self, conf):
        super().__init__(conf)
        self.conf = conf    
        self.interval = 9
        self.k_list = [7, 7]
        self.padding_num = 3
        self.s_list = [3, 3]
        
        self.feature_encoder = CNNEncoder(c_list=[6, 32, 64], k_list=self.k_list, s_list=self.s_list, p_list=[3,3])# (N,F/8,64)
        self.ori_encoder = CNNEncoder(c_list=[3, 32, 64], k_list=self.k_list, s_list=self.s_list, p_list=[3,3])# (N,F/8,64)
        self.leftIMU_encoder = CNNEncoder(c_list=[6, 32, 64], k_list=self.k_list, s_list=self.s_list, p_list=[3,3])# (N,F/8,64)
        self.gru1 = nn.GRU(input_size = 64, hidden_size =64, num_layers = 1, batch_first = True,bidirectional=True)
        self.gru2 = nn.GRU(input_size = 128, hidden_size =128, num_layers = 1, batch_first = True,bidirectional=True)
        self.fcn1 = nn.Sequential(nn.Linear(128, 128))
        self.batchnorm1 = torch.nn.BatchNorm1d(128)
        self.fcn2 = nn.Sequential(nn.Linear(128+64, 64))
        self.batchnorm2 = torch.nn.BatchNorm1d(64)
        
        self.gelu = nn.GELU()
        self.veldecoder = nn.Sequential(nn.Linear(256, 128),nn.GELU(), nn.Linear(128, 3))
        self.velcov_decoder = nn.Sequential(nn.Linear(256, 128),nn.GELU(), nn.Linear(128, 3))

    def encoder(self, feature, ori, leftIMU_feature):
        x1 = self.feature_encoder(feature.transpose(-1, -2)).transpose(-1, -2)
        x2 = self.ori_encoder(ori.transpose(-1,-2)).transpose(-1, -2) 
        x3 = self.leftIMU_encoder(leftIMU_feature.transpose(-1,-2)).transpose(-1, -2)
        x = torch.cat([x1, x2, x3], dim = -1)
        
        x = self.fcn2(x)
        x = self.batchnorm2(x.transpose(-1,-2)).transpose(-1,-2)
        x = self.gelu(x)
        x, _ = self.gru1(x)
        x, _ = self.gru2(x)
        return x
    
    def forward(self, data, rot=None):
        assert rot is not None
        feature = torch.cat([data["acc"], data["gyro"]], dim = -1)
        leftIMU_feature = torch.cat([data["acc_left"], data["gyro_left"]], dim = -1)
        feature = self.encoder(feature, rot, leftIMU_feature)
        net_vel = self.decoder(feature)
   
        #covariance propagation
        cov = None
        if self.conf.propcov:
            cov = self.cov_decoder(feature)
        return {"cov": cov, 'net_vel': net_vel}

class CodeNetMotionwithRot_Combined(CodeNetMotion):
    def __init__(self, conf):
        super().__init__(conf)
        self.conf = conf    
        self.interval = 9
        self.k_list = [7, 7]
        self.padding_num = 3
        self.s_list = [3, 3]
        
        self.feature_encoder = CNNEncoder(c_list=[6, 32, 64], k_list=self.k_list, s_list=self.s_list, p_list=[3,3])# (N,F/8,64)
        self.ori_encoder = CNNEncoder(c_list=[3, 32, 64], k_list=self.k_list, s_list=self.s_list, p_list=[3,3])# (N,F/8,64)
        self.baro_encoder = CNNEncoder(c_list=[1, 32, 32], k_list=self.k_list, s_list=self.s_list, p_list=[3,3])# (N,F/8,32)
        self.mag_encoder = CNNEncoder(c_list=[1, 32, 32], k_list=self.k_list, s_list=self.s_list, p_list=[3,3])# (N,F/8,32)
        self.joint_encoder = CNNEncoder(c_list=[6*9, 64, 32], k_list=self.k_list, s_list=self.s_list, p_list=[3,3])# (N,F/8,32)
        self.leftIMU_encoder = CNNEncoder(c_list=[6, 32, 32], k_list=self.k_list, s_list=self.s_list, p_list=[3,3])# (N,F/8,32)
        self.gru1 = nn.GRU(input_size = 64, hidden_size =64, num_layers = 1, batch_first = True,bidirectional=True)
        self.gru2 = nn.GRU(input_size = 128, hidden_size =128, num_layers = 1, batch_first = True,bidirectional=True)
        self.fcn1 = nn.Sequential(nn.Linear(128, 128))
        self.batchnorm1 = torch.nn.BatchNorm1d(128)
        self.fcn2 = nn.Sequential(nn.Linear(128 + 32*4, 64))
        self.batchnorm2 = torch.nn.BatchNorm1d(64)
        
        self.gelu = nn.GELU()
        self.veldecoder = nn.Sequential(nn.Linear(256, 128),nn.GELU(), nn.Linear(128, 3))
        self.velcov_decoder = nn.Sequential(nn.Linear(256, 128),nn.GELU(), nn.Linear(128, 3))

    def keep_legs_only(self, joint_rot, include_pelvis=False):
        # joint_rot: [B, K, 24*6]
        B, K, _ = joint_rot.shape
        jr = joint_rot.view(B, K, 24, 6)
        leg_idx = [1,4,7,10,  2,5,8,11]  # L hip/knee/ankle/foot, R hip/knee/ankle/foot
        if include_pelvis:
            leg_idx = [0] + leg_idx
        keep = torch.tensor(leg_idx, device=joint_rot.device, dtype=torch.long)

        jr_kept = torch.index_select(jr, dim=2, index=keep).reshape(B, K, keep.numel()*6)
        return jr_kept, keep

    def encoder(self, feature, ori, baro, mag, leftIMU_feature, joint_rot):
        x1 = self.feature_encoder(feature.transpose(-1, -2)).transpose(-1, -2)
        x2 = self.ori_encoder(ori.transpose(-1,-2)).transpose(-1, -2) 
        x3 = self.baro_encoder(baro.transpose(-1,-2)).transpose(-1, -2)
        x4 = self.mag_encoder(mag.transpose(-1,-2)).transpose(-1, -2)
        x5 = self.leftIMU_encoder(leftIMU_feature.transpose(-1,-2)).transpose(-1, -2)
        joint_rot, kept_idx = self.keep_legs_only(joint_rot, include_pelvis=True)
        x6 = self.joint_encoder(joint_rot.transpose(-1,-2)).transpose(-1, -2)
        x = torch.cat([x1, x2, x3, x4, x5, x6], dim = -1)
        
        x = self.fcn2(x)
        x = self.batchnorm2(x.transpose(-1,-2)).transpose(-1,-2)
        x = self.gelu(x)
        x, _ = self.gru1(x)
        x, _ = self.gru2(x)
        return x
    
    def forward(self, data, rot=None, joint_rot=None):
        assert rot is not None
        feature = torch.cat([data["acc"], data["gyro"]], dim = -1)
        leftIMU_feature = torch.cat([data["acc_left"], data["gyro_left"]], dim = -1)
        feature = self.encoder(feature, rot, data["baro"], data["mag"], leftIMU_feature, joint_rot=joint_rot)
        net_vel = self.decoder(feature)
   
        #covariance propagation
        cov = None
        if self.conf.propcov:
            cov = self.cov_decoder(feature)
        return {"cov": cov, 'net_vel': net_vel}

class CodeNetMotionwithRot_Combined_Gated_fusion(CodeNetMotion):
    def __init__(self, conf):
        super().__init__(conf)
        self.conf = conf    
        self.interval = 9
        self.k_list = [7, 7]
        self.padding_num = 3
        self.s_list = [3, 3]

        # ---------------- Encoders (same as yours) ----------------
        self.feature_encoder = CNNEncoder(c_list=[6, 32, 64], k_list=self.k_list, s_list=self.s_list, p_list=[3,3])  # -> 64
        self.ori_encoder     = CNNEncoder(c_list=[3, 32, 64], k_list=self.k_list, s_list=self.s_list, p_list=[3,3])  # -> 64
        self.baro_encoder    = CNNEncoder(c_list=[1, 32, 32], k_list=self.k_list, s_list=self.s_list, p_list=[3,3])  # -> 32
        self.mag_encoder     = CNNEncoder(c_list=[1, 32, 32], k_list=self.k_list, s_list=self.s_list, p_list=[3,3])  # -> 32
        self.joint_encoder   = CNNEncoder(c_list=[6*9, 64, 32], k_list=self.k_list, s_list=self.s_list, p_list=[3,3])# -> 32
        self.leftIMU_encoder = CNNEncoder(c_list=[6, 32, 32], k_list=self.k_list, s_list=self.s_list, p_list=[3,3])  # -> 32

        # ---------------- Gated Fusion (NEW) ----------------
        self.fuse_dim = 64
        self.num_mod = 6

        # project each modality to same dim=64
        # x1(64)->64, x2(64)->64, x3/4/5/6(32)->64
        self.mod_projs = nn.ModuleList([
            nn.Linear(64, self.fuse_dim),  # feature
            nn.Linear(64, self.fuse_dim),  # ori
            nn.Linear(32, self.fuse_dim),  # baro
            nn.Linear(32, self.fuse_dim),  # mag
            nn.Linear(32, self.fuse_dim),  # leftIMU
            nn.Linear(32, self.fuse_dim),  # joint
        ])

        # gating network outputs weights over modalities per timestep: [B,T,6]
        self.gate_net = nn.Sequential(
            nn.Linear(self.num_mod * self.fuse_dim, 128),
            nn.GELU(),
            nn.Linear(128, self.num_mod),
        )

        # optional: modality dropout prob (set 0.0 to disable)
        self.mod_dropout_p = getattr(conf, "mod_dropout_p", 0.0)

        # ---------------- Backbones (mostly same) ----------------
        self.gru1 = nn.GRU(input_size=64, hidden_size=64, num_layers=1, batch_first=True, bidirectional=True)
        self.gru2 = nn.GRU(input_size=128, hidden_size=128, num_layers=1, batch_first=True, bidirectional=True)

        # you can keep these if other code uses them, but fcn2 is no longer needed for fusion
        self.fcn1 = nn.Sequential(nn.Linear(128, 128))
        self.batchnorm1 = torch.nn.BatchNorm1d(128)

        # self.fcn2 = nn.Sequential(nn.Linear(128 + 32*4, 64))  # (OLD) no longer used
        self.batchnorm2 = torch.nn.BatchNorm1d(64)

        self.gelu = nn.GELU()
        self.veldecoder = nn.Sequential(nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 3))
        self.velcov_decoder = nn.Sequential(nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 3))

    def keep_legs_only(self, joint_rot, include_pelvis=False):
        B, K, _ = joint_rot.shape
        jr = joint_rot.view(B, K, 24, 6)
        leg_idx = [1,4,7,10,  2,5,8,11]
        if include_pelvis:
            leg_idx = [0] + leg_idx
        keep = torch.tensor(leg_idx, device=joint_rot.device, dtype=torch.long)
        jr_kept = torch.index_select(jr, dim=2, index=keep).reshape(B, K, keep.numel()*6)
        return jr_kept, keep

    def gated_fuse(self, feats):
        """
        feats: list of modality features [x1..x6], each is [B,T,Dm]
        returns: fused [B,T,64], weights [B,T,6]
        """
        proj_feats = []
        for i, x in enumerate(feats):
            proj_feats.append(self.mod_projs[i](x))  # -> [B,T,64]

        # stack: [B,T,M,64]
        P = torch.stack(proj_feats, dim=2)

        # optional modality dropout (training only)
        if self.training and self.mod_dropout_p > 0.0:
            mask = (torch.rand(P.shape[0], P.shape[1], P.shape[2], device=P.device) > self.mod_dropout_p).float()
            P = P * mask.unsqueeze(-1)

        # gating input: concat all projected feats: [B,T,6*64]
        g_in = torch.cat(proj_feats, dim=-1)
        w = torch.softmax(self.gate_net(g_in), dim=-1)  # [B,T,6]

        # weighted sum: [B,T,64]
        fused = (w.unsqueeze(-1) * P).sum(dim=2)
        return fused, w

    def encoder(self, feature, ori, baro, mag, leftIMU_feature, joint_rot):
        x1 = self.feature_encoder(feature.transpose(-1, -2)).transpose(-1, -2)       # [B,T,64]
        x2 = self.ori_encoder(ori.transpose(-1,-2)).transpose(-1, -2)                # [B,T,64]
        x3 = self.baro_encoder(baro.transpose(-1,-2)).transpose(-1, -2)              # [B,T,32]
        x4 = self.mag_encoder(mag.transpose(-1,-2)).transpose(-1, -2)                # [B,T,32]
        x5 = self.leftIMU_encoder(leftIMU_feature.transpose(-1,-2)).transpose(-1, -2)# [B,T,32]

        joint_rot, kept_idx = self.keep_legs_only(joint_rot, include_pelvis=True)
        x6 = self.joint_encoder(joint_rot.transpose(-1,-2)).transpose(-1, -2)        # [B,T,32]

        # ---------------- NEW: gated fusion ----------------
        x, w = self.gated_fuse([x1, x2, x3, x4, x5, x6])  # x: [B,T,64], w: [B,T,6]

        # keep your normalization + GRUs
        x = self.batchnorm2(x.transpose(-1, -2)).transpose(-1, -2)
        x = self.gelu(x)
        x, _ = self.gru1(x)
        x, _ = self.gru2(x)
        return x, w

    def forward(self, data, rot=None, joint_rot=None):
        assert rot is not None
        feature = torch.cat([data["acc"], data["gyro"]], dim=-1)
        leftIMU_feature = torch.cat([data["acc_left"], data["gyro_left"]], dim=-1)

        feat, gate_w = self.encoder(feature, rot, data["baro"], data["mag"], leftIMU_feature, joint_rot=joint_rot)
        net_vel = self.decoder(feat)  # or self.veldecoder(feat) depending on your base class

        cov = None
        if self.conf.propcov:
            cov = self.cov_decoder(feat)

        return {"cov": cov, "net_vel": net_vel, "gate_w": gate_w}
