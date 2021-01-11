import torch
import torch.nn as nn
from .arches import conv3x3, conv5x5, ResBlock
from thop import profile


class RNNCell(nn.Module):
    def __init__(self, dual_cell=True):
        super(RNNCell, self).__init__()
        self.dual_cell = dual_cell
        # F_B: blur feature extraction part
        self.F_B = nn.Sequential(
            conv5x5(3, 20, stride=1),
            conv5x5(20, 40, stride=2),
            conv5x5(40, 60, stride=2)
        )
        # F_R: residual blocks part
        res_blocks = []
        for i in range(6):
            res_blocks.append(ResBlock(80, batch_norm=False))
        self.F_R = nn.Sequential(*res_blocks)
        if not dual_cell:
            # F_L: reconstruct part
            self.F_L = nn.Sequential(
                nn.ConvTranspose2d(80, 40, 3, stride=2, padding=1, output_padding=1),
                nn.ConvTranspose2d(40, 20, 3, stride=2, padding=1, output_padding=1),
                conv5x5(20, 3, stride=1)
            )
        # F_h: hidden state part
        self.F_h = nn.Sequential(
            conv3x3(80, 20),
            ResBlock(20, batch_norm=False),
            conv3x3(20, 20)
        )

    def forward(self, x, h_last, infer=True):
        # x structure: (batch_size, channel, height, width)
        h = self.F_B(x)
        h = torch.cat([h, h_last], dim=1)  # Cat in channel dimension
        h = self.F_R(h)
        if not self.dual_cell and infer:
            out = self.F_L(h)
        else:
            out = None
        hc = self.F_h(h)
        return out, hc


class Model(nn.Module):
    """
    Recurrent Neural Networks with Intra-Frame Iterations for Video Deblurring (IFIRNN, CVPR2019)
    """

    def __init__(self, para):
        super(Model, self).__init__()
        self.para = para
        self.ratio = 4
        # C2H3
        self.iters = 3
        self.rnncell0 = RNNCell(dual_cell=True)
        self.rnncell1 = RNNCell(dual_cell=False)

    def forward(self, x, profile_flag=False):
        outputs = []
        # x structure: (batch_size, frame, channel, height, width) = (64, 12, 3, 720, 1024)
        batch_size, frames, channels, height, width = x.shape
        h_height = int(height / self.ratio)
        h_width = int(width / self.ratio)
        # forward h structure: (batch_size, channel, height, width)
        hc = torch.zeros(batch_size, 20, h_height, h_width).cuda()
        for i in range(frames):
            # output: (batch_size, channel, height, width) = (64, 3, 720, 1204)
            out, hc = self.rnncell0(x[:, i, :, :, :], hc)
            assert out == None
            for j in range(self.iters):
                if j == self.iters - 1:
                    out, hc = self.rnncell1(x[:, i, :, :, :], hc)
                else:
                    out, hc = self.rnncell1(x[:, i, :, :, :], hc, infer=False)
                    assert out == None
            outputs.append(torch.unsqueeze(out, dim=1))

        return torch.cat(outputs, dim=1)


def feed(model, iter_samples):
    inputs = iter_samples[0]
    outputs = model(inputs)
    return outputs


def cost_profile(model, H, W, seq_length):
    x = torch.randn(1, seq_length, 3, H, W).cuda()
    profile_flag = True
    flops, params = profile(model, inputs=(x, profile_flag), verbose=False)

    return flops / seq_length, params
