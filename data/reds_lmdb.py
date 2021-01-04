import pickle
import random
from os.path import join

import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from .utils import Crop, Flip, ToTensor, normalize


class DeblurDataset(Dataset):
    def __init__(self, path, frames, future_frames, past_frames, crop_size=(256, 256), ds_type='train', centralize=True,
                 normalize=True):
        ds_name = 'reds'
        self.datapath_blur = join(path, '{}_{}'.format(ds_name, ds_type))
        self.datapath_gt = join(path, '{}_{}_gt'.format(ds_name, ds_type))
        with open(join(path, '{}_info_{}.pkl'.format(ds_name, ds_type)), 'rb') as f:
            self.seqs_info = pickle.load(f)
        self.transform = transforms.Compose([Crop(crop_size), Flip(), ToTensor()])
        self.frames = frames
        self.crop_h, self.crop_w = crop_size
        self.W = 1280
        self.H = 720
        self.C = 3
        self.num_ff = future_frames
        self.num_pf = past_frames
        self.normalize = normalize
        self.centralize = centralize
        self.env_blur = lmdb.open(self.datapath_blur, map_size=1099511627776)
        self.env_gt = lmdb.open(self.datapath_gt, map_size=1099511627776)
        self.txn_blur = self.env_blur.begin()
        self.txn_gt = self.env_gt.begin()

    def __getitem__(self, idx):
        idx += 1
        ori_idx = idx
        seq_idx, frame_idx = 0, 0
        blur_imgs, sharp_imgs = list(), list()
        for i in range(self.seqs_info['num']):
            seq_length = self.seqs_info[i]['length'] - self.frames + 1
            if idx - seq_length <= 0:
                seq_idx = i
                frame_idx = idx - 1
                break
            else:
                idx -= seq_length

        top = random.randint(0, self.H - self.crop_h)
        left = random.randint(0, self.W - self.crop_w)
        flip_lr_flag = random.randint(0, 1)
        flip_ud_flag = random.randint(0, 1)
        sample = {'top': top, 'left': left, 'flip_lr': flip_lr_flag, 'flip_ud': flip_ud_flag}

        for i in range(self.frames):
            try:
                blur_img, sharp_img = self.get_img(seq_idx, frame_idx + i, sample)
                blur_imgs.append(blur_img)
                sharp_imgs.append(sharp_img)
            except TypeError as err:
                print('Handling run-time error:', err)
                print('failed case: idx {}, seq_idx {}, frame_idx {}'.format(ori_idx, seq_idx, frame_idx))
        blur_imgs = torch.cat(blur_imgs, dim=0)
        sharp_imgs = torch.cat(sharp_imgs[self.num_pf:self.frames - self.num_ff], dim=0)
        return blur_imgs, sharp_imgs

    def get_img(self, seq_idx, frame_idx, sample):
        code = '%03d_%08d' % (seq_idx, frame_idx)
        code = code.encode()
        blur_img = self.txn_blur.get(code)
        blur_img = np.frombuffer(blur_img, dtype='uint8')
        blur_img = blur_img.reshape(self.H, self.W, self.C)
        sharp_img = self.txn_gt.get(code)
        sharp_img = np.frombuffer(sharp_img, dtype='uint8')
        sharp_img = sharp_img.reshape(self.H, self.W, self.C)
        sample['image'] = blur_img
        sample['label'] = sharp_img
        sample = self.transform(sample)
        blur_img = normalize(sample['image'], centralize=self.centralize, normalize=self.normalize)
        sharp_img = normalize(sample['label'], centralize=self.centralize, normalize=self.normalize)

        return blur_img, sharp_img

    def __len__(self):
        return self.seqs_info['length'] - (self.frames - 1) * self.seqs_info['num']


class Dataloader:
    def __init__(self, para, device_id, ds_type='train'):
        path = join(para.data_root, para.dataset)
        dataset = DeblurDataset(path, para.frames, para.future_frames, para.past_frames, para.patch_size, ds_type,
                                para.centralize, para.normalize)
        gpus = para.num_gpus
        bs = para.batch_size
        ds_len = len(dataset)
        if para.trainer_mode == 'ddp':
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=para.num_gpus,
                rank=device_id
            )
            self.loader = DataLoader(
                dataset=dataset,
                batch_size=para.batch_size,
                shuffle=False,
                num_workers=para.threads,
                pin_memory=True,
                sampler=sampler
            )
            loader_len = np.ceil(ds_len / gpus)
            self.loader_len = int(np.ceil(loader_len / bs) * bs)

        elif para.trainer_mode == 'dp':
            self.loader = DataLoader(
                dataset=dataset,
                batch_size=para.batch_size,
                shuffle=True,
                num_workers=para.threads,
                pin_memory=True
            )
            self.loader_len = int(np.ceil(ds_len / bs) * bs)

    def __iter__(self):
        return iter(self.loader)

    def __len__(self):
        return self.loader_len
