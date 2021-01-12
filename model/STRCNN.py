import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .arches import conv3x3, conv5x5, deconv4x4
from thop import profile


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm=True):
        super(ResBlock, self).__init__()
        op = []
        for i in range(2):
            op.append(conv3x3(in_channels, out_channels))
            if batch_norm:
                op.append(nn.BatchNorm2d(out_channels))
            if i == 0:
                op.append(nn.ReLU(inplace=True))
        self.main_branch = nn.Sequential(*op)

    def forward(self, x):
        out = self.main_branch(x)
        out += x
        out = F.relu(out)
        return out


class Encoder(nn.Module):
    def __init__(self, para):
        super(Encoder, self).__init__()
        self.future_frames = 2
        self.past_frames = 2
        c = 3
        self.m1 = nn.Sequential(
            conv5x5(c * (self.future_frames + 1 + self.past_frames), 64),
            nn.ReLU(True),
            conv3x3(64, 32, stride=2),
            nn.ReLU(True)
        )
        num_blocks = 4
        ops = []
        for i in range(num_blocks):
            ops.append(ResBlock(in_channels=64, out_channels=64))
        self.m2 = nn.Sequential(*ops)

    def forward(self, x, f_last):
        # x: (n,c,h,w) e.g. (4,3*5,256,256)
        h = self.m1(x)
        h = torch.cat((h, f_last), dim=1)
        h = self.m2(h)

        return h


class DTB(nn.Module):
    """
    Online Video Deblurring via Dynamic Temporal Blending Network (STRCNN+DTB, ICCV2017)
    """

    def __init__(self, para):
        super(DTB, self).__init__()
        self.filter = nn.Conv2d(128, 64, kernel_size=5, stride=1, bias=True)
        # after filter: (n,c,h-4,w-4)
        self.bias = Variable(torch.zeros(1), requires_grad=True).cuda()
        self.pad = nn.ConstantPad2d(2, 0)

    def forward(self, x, h_last):
        w = torch.cat((x, h_last), dim=1)
        w = self.filter(w)
        w = 2 * torch.abs(torch.sigmoid(w) - 0.5) + self.bias
        w = w.clamp(0, 1)
        w = self.pad(w)
        h = w * x + (1 - w) * h_last

        return h


class Decoder(nn.Module):
    def __init__(self, para):
        super(Decoder, self).__init__()
        num_blcoks = 4
        ops = []
        for i in range(num_blcoks):
            ops.append(ResBlock(in_channels=64, out_channels=64))
        self.m1 = nn.Sequential(*ops)
        self.m2 = nn.Sequential(
            deconv4x4(64, 64),
            nn.ReLU(True),
            conv3x3(64, 3)
        )
        self.f = nn.Sequential(
            conv3x3(64, 32),
            nn.ReLU(True)
        )

    def forward(self, x):
        l = self.m1(x)
        f = self.f(l)
        l = self.m2(l)

        return l, f


class Model(nn.Module):
    def __init__(self, para=None):
        super(Model, self).__init__()
        self.para = para
        assert para.future_frames == 2 and para.past_frames == 2, "STRCNN+DTB, m=2"
        self.future_frames = para.future_frames
        self.past_frames = para.past_frames
        self.encoder = Encoder(para)
        self.dtb = DTB(para)
        self.decoder = Decoder(para)

    def forward(self, x, profile_flag=False):
        # x: (n,f,c,h,w) e.g. (4,14,3,256,256)
        N, F, C, H, W = x.shape
        h_last = torch.zeros(N, 64, H // 2, W // 2).cuda()
        f_last = torch.zeros(N, 32, H // 2, W // 2).cuda()
        outputs = []
        for i in range(F - self.future_frames - self.past_frames):
            inputs = x[:, i:i + self.past_frames + self.future_frames + 1]
            inputs = inputs.reshape(N, -1, H, W)
            h = self.encoder(inputs, f_last)
            h = self.dtb(h, h_last)
            l, f = self.decoder(h)
            outputs.append(l.unsqueeze(dim=1))
            h_last = h
            f_last = f
        out = torch.cat(outputs, dim=1)
        return out


def feed(model, iter_samples):
    inputs = iter_samples[0]
    outputs = model(inputs)
    return outputs


def cost_profile(model, H, W, seq_length):
    x = torch.randn(1, seq_length, 3, H, W).cuda()
    profile_flag = True
    flops, params = profile(model, inputs=(x, profile_flag), verbose=False)

    return flops, params
