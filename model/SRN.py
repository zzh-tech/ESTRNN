import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile
from functools import partial
from .arches import conv5x5, actFunc, make_blocks, deconv5x5, CLSTM_cell


class ResBlock(nn.Module):
    """
    Residual block
    """

    def __init__(self, in_chs, activation='relu', batch_norm=False):
        super(ResBlock, self).__init__()
        op = []
        for i in range(2):
            op.append(conv5x5(in_chs, in_chs))
            if batch_norm:
                op.append(nn.BatchNorm2d(in_chs))
            if i == 0:
                op.append(actFunc(activation))
        self.main_branch = nn.Sequential(*op)

    def forward(self, x):
        out = self.main_branch(x)
        out += x
        return out


class EBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, act='relu'):
        super(EBlock, self).__init__()
        self.conv = nn.Sequential(conv5x5(in_channels, out_channels, stride), actFunc(act))
        self.resblock_stack = make_blocks(ResBlock, num_basic_block=3, in_chs=out_channels)

    def forward(self, x):
        out = self.conv(x)
        out = self.resblock_stack(out)
        return out


class DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, act='relu'):
        super(DBlock, self).__init__()
        self.resblock_stack = make_blocks(ResBlock, num_basic_block=3, in_chs=in_channels)
        self.deconv = nn.Sequential(deconv5x5(in_channels, out_channels, stride), actFunc(act))

    def forward(self, x):
        out = self.resblock_stack(x)
        out = self.deconv(out)
        return out


class OutBlock(nn.Module):
    def __init__(self, in_channels):
        super(OutBlock, self).__init__()
        self.resblock_stack = make_blocks(ResBlock, num_basic_block=3, in_chs=in_channels)
        self.conv = conv5x5(in_channels=in_channels, out_channels=3)

    def forward(self, x):
        out = self.resblock_stack(x)
        out = self.conv(out)
        return out


class Model(nn.Module):
    """
    Scale-recurrent Network for Deep Image Deblurring (SRN-Deblur, CVPR2018)
    """

    def __init__(self, para):
        super(Model, self).__init__()
        assert para.future_frames == 0 and para.past_frames == 0
        self.upsample_fn = partial(torch.nn.functional.interpolate, mode='bilinear', align_corners=False)
        self.inblock = EBlock(3 + 3, 32, 1)
        self.eblock1 = EBlock(32, 64, 2)
        self.eblock2 = EBlock(64, 128, 2)
        self.convlstm = CLSTM_cell(128, 128, 5)
        self.dblock1 = DBlock(128, 64, 2)
        self.dblock2 = DBlock(64, 32, 2)
        self.outblock = OutBlock(32)

        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                torch.nn.init.xavier_normal_(m.weight)

    def _forward_step(self, x, hidden_state):
        e32 = self.inblock(x)
        e64 = self.eblock1(e32)
        e128 = self.eblock2(e64)
        h, c = self.convlstm(e128, hidden_state)
        d64 = self.dblock1(c)
        d32 = self.dblock2(d64 + e64)
        d3 = self.outblock(d32 + e32)
        return d3, h, c

    def _forward_single(self, b1, b2, b3):
        h, c = self.convlstm.init_hidden(b3.shape[0], (b3.shape[-2] // 4, b3.shape[-1] // 4))
        i3, h, c = self._forward_step(torch.cat([b3, b3.clone().detach()], dim=1), (h, c))
        h = self.upsample_fn(h, scale_factor=2)
        c = self.upsample_fn(c, scale_factor=2)
        i2, h, c = self._forward_step(torch.cat([b2, self.upsample_fn(i3, scale_factor=2)], dim=1), (h, c))
        h = self.upsample_fn(h, scale_factor=2)
        c = self.upsample_fn(c, scale_factor=2)
        i1, h, c = self._forward_step(torch.cat([b1, self.upsample_fn(i2, scale_factor=2)], dim=1), (h, c))

        return i1, i2, i3

    def forward(self, x, profile_flag=False):
        b1 = x
        B, N, C, H, W = b1.shape
        b1 = b1.reshape(B, N * C, H, W)
        b2 = F.interpolate(b1, size=(H // 2, W // 2), mode='bilinear', align_corners=False)
        b3 = F.interpolate(b1, size=(H // 4, W // 4), mode='bilinear', align_corners=False)
        i1, i2, i3 = [], [], []
        for i in range(N):
            sub_b1, sub_b2, sub_b3 = b1[:, i * C:(i + 1) * C, :, :], b2[:, i * C:(i + 1) * C, :, :], b3[:,
                                                                                                     i * C:(i + 1) * C,
                                                                                                     :, :]
            sub_i1, sub_i2, sub_i3 = self._forward_single(sub_b1, sub_b2, sub_b3)
            i1.append(sub_i1.unsqueeze(dim=1))
            i2.append(sub_i2.unsqueeze(dim=1))
            i3.append(sub_i3.unsqueeze(dim=1))
        i1 = torch.cat(i1, dim=1)
        i2 = torch.cat(i2, dim=1)
        i3 = torch.cat(i3, dim=1)

        return i1, i2, i3


def feed(model, iter_samples):
    inputs = iter_samples[0]
    outputs = model(inputs)

    return outputs  # (i1, i2, i3)


def cost_profile(model, H, W, seq_length):
    assert seq_length == 1
    x = torch.randn(1, seq_length, 3, H, W).cuda()
    profile_flag = True
    flops, params = profile(model, inputs=(x, profile_flag), verbose=False)

    return flops / seq_length, params
