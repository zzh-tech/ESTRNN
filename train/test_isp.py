import rawpy
import cv2
import os
import numpy as np
from PIL import Image
from os.path import join, basename, dirname
from metrics import psnr_calculate, ssim_calculate
from utils import AverageMeter, img2video


def raw2rgb(path):
    img_raw = cv2.imread(path, -1)
    Image.fromarray(img_raw, mode="I;16").save('saved.tiff')
    raw_buf = rawpy.imread('saved.tiff')
    os.remove('saved.tiff')
    img = raw_buf.postprocess(use_auto_wb=True, no_auto_bright=False, output_bps=16, user_black=0)
    img = (img.astype(np.float32) / 65535 * 255).astype(np.uint8)
    save_path = join(dirname(path), basename(path).replace('tiff', 'png'))
    cv2.imwrite(save_path, img[:, :, ::-1])

    return save_path


def main(path):
    dirs = os.listdir(path)
    frame_start = 2
    frame_end = 98
    H, W = 480, 640
    PSNR = AverageMeter()
    SSIM = AverageMeter()
    marks = ['Input', 'ESTRNN-RAW', 'GT']

    for dir in dirs:
        if dir.endswith('.avi'):
            continue
        dir_path = join(path, dir)

        for i in range(frame_start, frame_end):
            img_raw_path = join(dir_path, '{:08d}_{}.tiff'.format(i, 'input'))
            _ = raw2rgb(img_raw_path)
            img_raw_path = join(dir_path, '{:08d}_{}.tiff'.format(i, 'estrnn-raw'))
            img_deblur_path = raw2rgb(img_raw_path)
            img_deblur = cv2.imread(img_deblur_path)
            img_raw_path = join(dir_path, '{:08d}_{}.tiff'.format(i, 'gt'))
            img_gt_path = raw2rgb(img_raw_path)
            img_gt = cv2.imread(img_gt_path)
            PSNR.update(psnr_calculate(img_deblur, img_gt))
            SSIM.update(ssim_calculate(img_deblur, img_gt))

        img2video(path=path, size=(3 * W, 1 * H), seq=dir, frame_start=frame_start, frame_end=frame_end,
                  marks=marks, fps=10)

    print('Test PSNR : {}'.format(PSNR.avg))
    print('Test SSIM : {}'.format(SSIM.avg))


if __name__ == '__main__':
    path = '../experiment/ESTRNN_RAW_2ms16ms_3e-4/BSD_ESTRNN-RAW_test/'
    main(path)
