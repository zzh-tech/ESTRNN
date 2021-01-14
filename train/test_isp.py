import rawpy
import cv2
import os
import numpy as np
from PIL import Image
from os.path import join
from metrics import psnr_calculate, ssim_calculate
from utils import AverageMeter, img2video


def raw2rgb(img_raw):
    import os
    Image.fromarray(img_raw, mode="I;16").save('saved.tiff')
    raw_buf = rawpy.imread('saved.tiff')
    os.remove('saved.tiff')
    img_rgb = raw_buf.postprocess(use_auto_wb=True, no_auto_bright=False, output_bps=16, user_black=0)
    img_rgb = (img_rgb.astype(np.float32) / 65535 * 255).astype(np.uint8)

    return img_rgb


def _main(path, ds_path):
    dirs = os.listdir(path)
    frame_start = 0
    frame_end = 150
    row_frames = 15
    H, W = 480, 640
    PSNR = AverageMeter()
    SSIM = AverageMeter()
    marks = ['Input', 'ESTRNN-RAW', 'GT']

    for dir in dirs:
        if dir.endswith('.avi'):
            continue
        dir_path = join(path, dir)

        imgs_deblur_raw = []

        for i in range((frame_end - frame_start) // row_frames):
            imgs_deblur_raw_row = []
            for j in range(row_frames):
                frame = i * row_frames + j
                if frame == frame_start + 0 or frame == frame_start + 1:
                    frame = frame_start + 2
                elif frame == frame_end - 1 or frame == frame_end - 2:
                    frame = frame_end - 3
                img_deblur_raw_path = join(dir_path, '{:08d}_{}.tiff'.format(frame, 'estrnn-raw'))
                img_deblur_raw = cv2.imread(img_deblur_raw_path, -1)
                assert not img_deblur_raw is None, 'i:{}, j:{}, frame:{}'.format(i, j, frame)
                imgs_deblur_raw_row.append(img_deblur_raw)
            imgs_deblur_raw_row = np.concatenate(imgs_deblur_raw_row, axis=1)
            assert imgs_deblur_raw_row.shape == (H, W * row_frames), imgs_deblur_raw_row.shape
            imgs_deblur_raw.append(imgs_deblur_raw_row)

        imgs_deblur_raw = np.concatenate(imgs_deblur_raw, axis=0)
        imgs_deblur = raw2rgb(imgs_deblur_raw)

        for i in range((frame_end - frame_start) // row_frames):
            for j in range(row_frames):
                frame = i * row_frames + j
                if frame in [frame_start + 0, frame_start + 1, frame_end - 1, frame_end - 2]:
                    continue
                img_deblur = imgs_deblur[i * H:(i + 1) * H, j * W:(j + 1) * W, ...][:, :, ::-1]
                save_path = join(dir_path, '{:08d}_{}.png'.format(frame, marks[1].lower()))
                cv2.imwrite(save_path, img_deblur)

                img_blur_path = join(ds_path, dir, 'Blur', 'RGB', '{:08d}.png'.format(frame))
                img_blur = cv2.imread(img_blur_path)
                save_path = join(dir_path, '{:08d}_{}.png'.format(frame, marks[0].lower()))
                cv2.imwrite(save_path, img_blur)

                img_gt_path = join(ds_path, dir, 'Sharp', 'RGB', '{:08d}.png'.format(frame))
                img_gt = cv2.imread(img_gt_path)
                save_path = join(dir_path, '{:08d}_{}.png'.format(frame, marks[2].lower()))
                cv2.imwrite(save_path, img_gt)

                PSNR.update(psnr_calculate(img_deblur, img_gt))
                SSIM.update(ssim_calculate(img_deblur, img_gt))

        img2video(path=path, size=(3 * W, 1 * H), seq=dir, frame_start=frame_start + 2, frame_end=frame_end - 2,
                  marks=marks, fps=10)

    print('Test PSNR : {}'.format(PSNR.avg))
    print('Test SSIM : {}'.format(SSIM.avg))


# def _isp_valid(path):
#     frame_start = 1
#     frame_end = 151
#     row_frames = 15
#     H, W = 480, 640
#     PSNR = AverageMeter()
#     SSIM = AverageMeter()
#     imgs_raw = []
#     for i in range((frame_end - frame_start) // row_frames):
#         imgs_raw_row = []
#         for j in range(row_frames):
#             frame = i * row_frames + j + 1
#             img_raw_path = join(path, 'Sharp', 'RAW', '{}.tiff'.format(frame))
#             img_raw = cv2.imread(img_raw_path, -1)
#             imgs_raw_row.append(img_raw)
#         imgs_row = np.concatenate(imgs_raw_row, axis=1)
#         assert imgs_row.shape == (H, W * row_frames), imgs_row.shape
#         imgs_raw.append(imgs_row)
#
#     imgs_raw = np.concatenate(imgs_raw, axis=0)
#     imgs = raw2rgb(imgs_raw)
#
#     for i in range((frame_end - frame_start) // row_frames):
#         for j in range(row_frames):
#             frame = i * row_frames + j + 1
#             img = imgs[i * H:(i + 1) * H, j * W:(j + 1) * W, ...][:, :, ::-1]
#             save_path = join(path, 'Sharp', 'RGB', '{}_{}.png'.format(frame, 'pyisp'))
#             cv2.imwrite(save_path, img)
#
#             img_gt_path = join(path, 'Sharp', 'RGB', '{}.png'.format(frame))
#             img_gt = cv2.imread(img_gt_path)
#
#             PSNR.update(psnr_calculate(img, img_gt))
#             SSIM.update(ssim_calculate(img, img_gt))
#
#     print('Test PSNR : {}'.format(PSNR.avg))
#     print('Test SSIM : {}'.format(SSIM.avg))


# def rebuild_test(ds_path, src_path):
#     import shutil
#     dirs = os.listdir(ds_path)
#     for dir in dirs:
#         dir_num = int(dir)
#         src_dir = '_Scene{}'.format(dir_num + 1)
#         src_dir_path = join(src_path, src_dir)
#         assert os.path.exists(src_dir_path), src_dir_path
#         for set_type in ['Blur', 'Sharp']:
#             for data_format in ['RAW', 'RGB']:
#                 sub_src_dir_path = join(src_dir_path, set_type, data_format)
#                 for i in range(101, 151):
#                     src_img = '{}.png'.format(i) if data_format == 'RGB' else '{}.tiff'.format(i)
#                     ds_img = '{:08d}.png'.format(i - 1) if data_format == 'RGB' else '{:08d}.tiff'.format(i - 1)
#                     src_img_path = join(sub_src_dir_path, src_img)
#                     assert os.path.exists(src_img_path)
#                     ds_img_path = join(ds_path, dir, set_type, data_format, ds_img)
#                     shutil.copy(src=src_img_path, dst=ds_img_path)


if __name__ == '__main__':
    path = '../results/ESTRNN_BSD_RAW_2ms16ms_3e-4/BSD_ESTRNN-RAW_test/'
    ds_path = '/home/zhong/Dataset/BSD/BSD_2ms16ms/test/'
    _main(path, ds_path)

    # valid_path = '/home/zhong/Desktop/scene16/'
    # _isp_valid(valid_path)

    # ds_config = '3ms24ms' # 1ms8ms | 2ms16ms | 3ms24ms
    # ds_path = '/home/zhong/Dataset/BSD/BSD_{}/test/'.format(ds_config)
    # src_path = '/home/zhong/Desktop/BSD/{}/'.format(ds_config)
    # rebuild_test(ds_path, src_path)
