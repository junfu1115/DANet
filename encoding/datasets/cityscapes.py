###########################################################################
# Created by: Hang Zhang
# Email: zhang.hang@rutgers.edu
# Copyright (c) 2017
###########################################################################

import os
import sys
import numpy as np
import random
from PIL import Image, ImageOps, ImageFilter

import torch
import re
import torch.utils.data as data
import torchvision.transforms as transform
from tqdm import tqdm
from .base import BaseDataset

class CitySegmentation(BaseDataset):
    NUM_CLASS = 19
    BASE_DIR = 'cityscapes'
    def __init__(self, root=os.path.expanduser('../../datasets'), split='train',
                 mode=None, transform=None, target_transform=None, **kwargs):
        super(CitySegmentation, self).__init__(
            root, split, mode, transform, target_transform, **kwargs)
        # assert exists and prepare dataset automatically
        #assert os.path.exists(root), "Please setup the dataset using" + \
         #   "encoding/scripts/cityscapes.py"
        root = os.path.join(root, self.BASE_DIR)
        assert os.path.exists(root), "Please download the dataset!!"
        self.images, self.masks = get_city_pairs(root, self.split)
        if split != 'test':
            assert (len(self.images) == len(self.masks))
        if len(self.images) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: \
                " + root + "\n"))

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        if self.mode == 'test':
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.images[index])
        
        mask = Image.open(self.masks[index])
        # synchrosized transform
        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'val':
            img, mask = self._val_sync_transform(img, mask)
        else:
            assert self.mode == 'testval'
            mask = self._mask_transform(mask)

        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            mask = self.target_transform(mask)

        return img, mask

    def _mask_transform(self, mask):
        target = np.array(mask).astype('int32')
        mapping_20 = {
            0: 255,
            1: 255,
            2: 255,
            3: 255,
            4: 255,
            5: 255,
            6: 255,
            7: 0,
            8: 1,
            9: 255,
            10: 255,
            11: 2,
            12: 3,
            13: 4,
            14: 255,
            15: 255,
            16: 255,
            17: 5,
            18: 255,
            19: 6,
            20: 7,
            21: 8,
            22: 9,
            23: 10,
            24: 11,
            25: 12,
            26: 13,
            27: 14,
            28: 15,
            29: 255,
            30: 255,
            31: 16,
            32: 17,
            33: 18,
            -1: 255,
        }

        label_mask = np.zeros_like(target)
        for k in mapping_20:
            label_mask[target == k] = mapping_20[k]
        label_mask[label_mask == 255] = -1
        return torch.from_numpy(label_mask).long()

    def __len__(self):
        return len(self.images)

    @property
    def pred_offset(self):
        return 0


def get_city_pairs(folder, split='train'):
    def get_path_pairs(folder,split_f):
        img_paths = []
        mask_paths = []
        with open(split_f, 'r') as lines:
            for line in tqdm(lines):
                ll_str = line.rstrip()
                imgpath = os.path.join(folder,'leftImg8bit/val', ll_str+'_leftImg8bit.png')
                maskpath = os.path.join(folder,'gtFine/val', ll_str+'_gtFine_labelIds.png')
                if os.path.isfile(maskpath):
                    img_paths.append(imgpath)
                    mask_paths.append(maskpath)
                else:
                    print('cannot find the mask:', maskpath)
        return img_paths, mask_paths
    if split == 'train':
        split_f = os.path.join(folder, 'train_fine.txt')
        img_paths, mask_paths = get_path_pairs(folder, split_f)
    elif split == 'val':
        split_f = os.path.join(folder, 'val.txt')
        img_paths, mask_paths = get_path_pairs(folder, split_f)
    elif split == 'test':
        split_f = os.path.join(folder, 'test.txt')
        img_paths, mask_paths = get_path_pairs(folder, split_f)
    else:
        split_f = os.path.join(folder, 'trainval_fine.txt')
        img_paths, mask_paths = get_path_pairs(folder, split_f)
    return img_paths, mask_paths
