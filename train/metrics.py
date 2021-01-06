# import lpips
import numpy as np
import torch
from skimage.metrics import structural_similarity as compare_ssim
from torch.nn.modules.loss import _Loss

from data.utils import normalize_reverse


def estimate_mask(img):
    mask = img.copy()
    mask[mask > 0.0] = 1.0
    return mask


def mask_pair(x, y, mask):
    return x * mask, y * mask


def im2tensor(image, cent=1., factor=255. / 2.):
    image = image.astype(np.float)
    return torch.Tensor((image / factor - cent)
                        [:, :, :, np.newaxis].transpose((3, 2, 0, 1)))


def psnr_calculate(x, y, val_range=255.0):
    # x,y size (h,w,c)
    # assert len(x.shape) == 3
    # assert len(y.shape) == 3
    x = x.astype(np.float)
    y = y.astype(np.float)
    diff = (x - y) / val_range
    mse = np.mean(diff ** 2)
    psnr = -10 * np.log10(mse)
    return psnr


def ssim_calculate(x, y, val_range=255.0):
    ssim = compare_ssim(y, x, multichannel=True, gaussian_weights=True, sigma=1.5, use_sample_covariance=False,
                        data_range=val_range)
    return ssim


# def lpips_calculate(x, y, net='alex', gpu=False):
#     # input range is 0~255
#     # image should be RGB, and normalized to [-1,1]
#     x = im2tensor(x[:, :, ::-1])
#     y = im2tensor(y[:, :, ::-1])
#     loss_fn = lpips.LPIPS(net=net, verbose=False)
#     if gpu:
#         x = x.cuda()
#         y = y.cuda()
#         loss_fn = loss_fn.cuda()
#     lpips_value = loss_fn(x, y)
#     return lpips_value.item()


class PSNR(_Loss):
    def __init__(self, centralize=True, normalize=True, val_range=255.):
        super(PSNR, self).__init__()
        self.centralize = centralize
        self.normalize = normalize
        self.val_range = val_range

    def _quantize(self, img):
        img = normalize_reverse(img, centralize=self.centralize, normalize=self.normalize, val_range=self.val_range)
        img = img.clamp(0, self.val_range).round()
        return img

    def forward(self, x, y):
        diff = self._quantize(x) - self._quantize(y)
        if x.dim() == 3:
            n = 1
        elif x.dim() == 4:
            n = x.size(0)
        elif x.dim() == 5:
            n = x.size(0) * x.size(1)

        mse = diff.div(self.val_range).pow(2).view(n, -1).mean(dim=-1)
        psnr = -10 * mse.log10()

        return psnr.mean()
