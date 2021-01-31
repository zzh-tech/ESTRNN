import os
import random
from os.path import join

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

try:
    from .utils import normalize, Crop, Flip, ToTensor
except:
    from utils import normalize, Crop, Flip, ToTensor


class DeblurDataset(Dataset):
    """
    Structure of self_.records:
        seq:
            frame:
                path of images -> {'Blur': <path>, 'Sharp': <path>}
    """

    def __init__(self, path, frames, future_frames, past_frames, crop_size=(256, 256), centralize=True, normalize=True):
        assert frames - future_frames - past_frames >= 1
        self.frames = frames
        self.num_ff = future_frames
        self.num_pf = past_frames
        self.W = 512
        self.H = 512
        self.crop_h, self.crop_w = crop_size
        self.normalize = normalize
        self.centralize = centralize
        self.transform = transforms.Compose([Crop(crop_size), Flip(), ToTensor()])
        self._samples = self._generate_samples(path)

    def _generate_samples(self, dataset_path):
        samples = list()
        records = dict()
        seqs = sorted(os.listdir(dataset_path))
        for seq in seqs:
            records[seq] = list()
            seq_path = join(dataset_path, seq)
            imgs_num = len(os.listdir(join(seq_path, 'Blur')))
            for i in range(imgs_num):
                sample = dict()
                sample['Blur'] = join(seq_path, 'Blur', '{:08d}.png'.format(i))
                sample['Sharp'] = join(seq_path, 'Sharp', '{:08d}.png'.format(i))
                records[seq].append(sample)
        for seq_records in records.values():
            temp_length = len(seq_records) - (self.frames - 1)
            if temp_length <= 0:
                raise IndexError('Exceed the maximum length of the video sequence')
            for idx in range(temp_length):
                samples.append(seq_records[idx:idx + self.frames])
        return samples

    def __getitem__(self, item):
        top = random.randint(0, self.H - self.crop_h)
        left = random.randint(0, self.W - self.crop_w)
        flip_lr = random.randint(0, 1)
        flip_ud = random.randint(0, 1)
        sample = {'top': top, 'left': left, 'flip_lr': flip_lr, 'flip_ud': flip_ud}

        blur_imgs, sharp_imgs = [], []
        for sample_dict in self._samples[item]:
            blur_img, sharp_img = self._load_sample(sample_dict, sample)
            blur_imgs.append(blur_img)
            sharp_imgs.append(sharp_img)
        sharp_imgs = sharp_imgs[self.num_pf:self.frames - self.num_ff]
        return [torch.cat(item, dim=0) for item in [blur_imgs, sharp_imgs]]

    def _load_sample(self, sample_dict, sample):
        sample['image'] = cv2.imread(sample_dict['Blur'])
        sample['label'] = cv2.imread(sample_dict['Sharp'])
        sample = self.transform(sample)
        blur_img = normalize(sample['image'], centralize=self.centralize, normalize=self.normalize)
        sharp_img = normalize(sample['label'], centralize=self.centralize, normalize=self.normalize)
        return blur_img, sharp_img

    def __len__(self):
        return len(self._samples)


class Dataloader:
    def __init__(self, para, device_id, ds_type='train'):
        path = join(para.data_root, para.dataset, ds_type)
        frames = para.frames
        dataset = DeblurDataset(path, frames, para.future_frames, para.past_frames, para.patch_size,
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
                sampler=sampler,
                drop_last=True
            )
            loader_len = np.ceil(ds_len / gpus)
            self.loader_len = int(np.ceil(loader_len / bs) * bs)

        elif para.trainer_mode == 'dp':
            self.loader = DataLoader(
                dataset=dataset,
                batch_size=para.batch_size,
                shuffle=True,
                num_workers=para.threads,
                pin_memory=True,
                drop_last=True
            )
            self.loader_len = int(np.ceil(ds_len / bs) * bs)

    def __iter__(self):
        return iter(self.loader)

    def __len__(self):
        return self.loader_len


if __name__ == '__main__':
    from para import Parameter

    para = Parameter().args
    para.dataset = 'syn_blur_30'
    dataloader = Dataloader(para, 0)
    for x, y in dataloader:
        print(x.shape, y.shape)
        break
    print(x.type(), y.type())
    print(np.max(x.numpy()), np.min(x.numpy()))
    print(np.max(y.numpy()), np.min(y.numpy()))
