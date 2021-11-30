import cv2
from argparse import ArgumentParser
from glob import glob
from os.path import join
from train.metrics import psnr_calculate, ssim_calculate
from train.utils import AverageMeter

if __name__ == '__main__':
    parser = ArgumentParser(description='BSD')
    parser.add_argument('--data_dir', type=str, required=True, help='where store the BSD dataset')
    args = parser.parse_args()

    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()
    val_range = 2.0 ** 8 - 1

    blur_img_paths = glob(join(args.data_dir, "**", 'Blur', 'RGB', '*.png'), recursive=True)
    sharp_img_paths = [img.replace('Blur', 'Sharp') for img in blur_img_paths]
    for blur_img_path, sharp_img_path in zip(blur_img_paths, sharp_img_paths):
        blur_img = cv2.imread(blur_img_path)
        sharp_img = cv2.imread(sharp_img_path)
        psnr_meter.update(psnr_calculate(blur_img, sharp_img, val_range=val_range))
        ssim_meter.update(ssim_calculate(blur_img, sharp_img, val_range=val_range))

    print('{} images, psnr: {}, ssim: {}'.format(psnr_meter.count, psnr_meter.avg, ssim_meter.avg))
