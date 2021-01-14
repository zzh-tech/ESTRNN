import torch
import torch.nn as nn
import torch.nn.functional as F
from .arches import conv3x3, conv5x5
from thop import profile


class Model(nn.Module):
    """
    Deep Video Deblurring for Hand-held Cameras (DBN, CVPR2017)
    """
    def __init__(self, para):
        super(Model, self).__init__()
        self.para = para
        assert para.future_frames == 2 and para.past_frames == 2, "DBN takes 5 consecutive frames as input"
        self.future_frames = para.future_frames
        self.past_frames = para.past_frames
        self.F0 = nn.Sequential(
            conv5x5(15, 64, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )
        self.D1 = nn.Sequential(
            conv3x3(64, 64, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )
        self.F1_1 = nn.Sequential(
            conv3x3(64, 128, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )
        self.F1_2 = nn.Sequential(
            conv3x3(128, 128, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )
        self.D2 = nn.Sequential(
            conv3x3(128, 256, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )
        self.F2_1 = nn.Sequential(
            conv3x3(256, 256, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )
        self.F2_2 = nn.Sequential(
            conv3x3(256, 256, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )
        self.F2_3 = nn.Sequential(
            conv3x3(256, 256, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )
        self.D3 = nn.Sequential(
            conv3x3(256, 512, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )
        self.F3_1 = nn.Sequential(
            conv3x3(512, 512, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )
        self.F3_2 = nn.Sequential(
            conv3x3(512, 512, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )
        self.F3_3 = nn.Sequential(
            conv3x3(512, 512, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )
        self.U1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256)
        )
        # skip connection 1 from F2_3
        # ReLU
        self.F4_1 = nn.Sequential(
            conv3x3(256, 256, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )
        self.F4_2 = nn.Sequential(
            conv3x3(256, 256, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )
        self.F4_3 = nn.Sequential(
            conv3x3(256, 256, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )
        self.U2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128)
        )
        # skip connection 2 from F1_2
        # ReLU
        self.F5_1 = nn.Sequential(
            conv3x3(128, 128, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )
        self.F5_2 = nn.Sequential(
            conv3x3(128, 64, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )
        self.U3 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64)
        )
        # skip connection 3 from F0
        # ReLU
        self.F6_1 = nn.Sequential(
            conv3x3(64, 15, stride=1),
            nn.BatchNorm2d(15),
            nn.ReLU(True)
        )
        self.F6_2 = nn.Sequential(
            conv3x3(15, 3, stride=1),
            nn.BatchNorm2d(3)
        )
        # skip connection 4 from input
        # Sigmoid

    def forward(self, x, profile_flag=False):
        # x: (n,f,c,h,w) | (4,10,3,256,256)
        future_frames = self.para.future_frames
        past_frames = self.para.past_frames
        num_subframes = future_frames + 1 + past_frames
        n, f, c, h, w = x.shape
        outputs = []
        for i in range(past_frames, f - future_frames):
            _x = x[:, i - past_frames:i + future_frames + 1]
            _x = _x.reshape(n, num_subframes * c, h, w)  # skip connection to F6_2
            # encoder and decoder
            out_F0 = self.F0(_x)  # skip connection to U3
            out_D1 = self.D1(out_F0)
            out_F1_1 = self.F1_1(out_D1)
            out_F1_2 = self.F1_2(out_F1_1)  # skip connection to U2
            out_D2 = self.D2(out_F1_2)
            out_F2_1 = self.F2_1(out_D2)
            out_F2_2 = self.F2_2(out_F2_1)
            out_F2_3 = self.F2_3(out_F2_2)  # skip connection to U1
            out_D3 = self.D3(out_F2_3)
            out_F3_1 = self.F3_1(out_D3)
            out_F3_2 = self.F3_2(out_F3_1)
            out_F3_3 = self.F3_3(out_F3_2)
            out_U1 = self.U1(out_F3_3)
            out_U1 = F.relu(out_U1 + out_F2_3)
            out_F4_1 = self.F4_1(out_U1)
            out_F4_2 = self.F4_2(out_F4_1)
            out_F4_3 = self.F4_3(out_F4_2)
            out_U2 = self.U2(out_F4_3)
            out_U2 = F.relu(out_U2 + out_F1_2)
            out_F5_1 = self.F5_1(out_U2)
            out_F5_2 = self.F5_2(out_F5_1)
            out_U3 = self.U3(out_F5_2)
            out_U3 = F.relu(out_U3 + out_F0)
            out_F6_1 = self.F6_1(out_U3)
            out_F6_2 = self.F6_2(out_F6_1)
            out_F6_2 = out_F6_2 + _x[:, (num_subframes // 2) * c:(num_subframes // 2) * c + c]
            out = out_F6_2.unsqueeze(dim=1)
            outputs.append(out)
        outputs = torch.cat(outputs, dim=1)

        return outputs


def feed(model, iter_samples):
    inputs = iter_samples[0]
    outputs = model(inputs)
    return outputs


def cost_profile(model, H, W, seq_length):
    x = torch.randn(1, seq_length, 3, H, W).cuda()
    profile_flag = True
    flops, params = profile(model, inputs=(x, profile_flag), verbose=False)

    return flops, params
