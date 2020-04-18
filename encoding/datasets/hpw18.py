# created by: Sean Liu
# Amazon Lab 126
from __future__ import print_function

import errno
import hashlib
import os
import sys
import tarfile
import numpy as np
import random
import math

import torch.utils.data as data
import PIL
from PIL import Image, ImageOps

from six.moves import urllib


class Segmentation_HPW18(data.Dataset):
    CLASSES = [
        'background', 'hat', 'hair', 'sunglasses', 'upper-clothes', 
        'skirt', 'pants', 'dress', 'belt', 'left-shoe', 'right-shoe', 
        'face', 'left-leg', 'right-leg', 'left-arm', 'right-arm', 'bag', 
        'scarf'
    ]

    URL = "/cvdata1/lliuqian/humanParsingDataset"
    FILE = "hpw18.tar.gz"
    MD5 = ''
    BASE_DIR = ''

    def __init__(self,
                 root,
                 train=True,
                 transform=None,
                 target_transform=None,
                 download=False):
        self.root = root
        _hpw18_root = os.path.join(self.root, self.BASE_DIR)
        _mask_dir = os.path.join(_hpw18_root, 'SegmentationClassAug_256x384')
        _image_dir = os.path.join(_hpw18_root, 'JPEGImages_256x384')
        self.transform = transform
        self.target_transform = target_transform
        self.train = train

        if download:
            self._download()

        # train/val/test splits are pre-cut
        _splits_dir = _hpw18_root
        _split_f = os.path.join(_splits_dir, 'humanparsingImageMask_256x384_absPath_train.txt')
        if not self.train:
            _split_f = os.path.join(_splits_dir, 'humanparsingImageMask_256x384_absPath_val.txt')

        print("reading from ", _split_f)

        self.images = []
        self.masks = []
        with open(os.path.join(_split_f), "r") as lines:
            for line in lines:
                s = line.split()
                _image = s[0] # image absolution path
                _mask = s[1] # mask absolution path
                assert os.path.isfile(_image)
                assert os.path.isfile(_mask)
                self.images.append(_image)
                self.masks.append(_mask)
        assert (len(self.images) == len(self.masks))

    def __getitem__(self, index):
        _img = Image.open(self.images[index]).convert('RGB')
        _timg = Image.open(self.masks[index])
        _target = np.array(_timg, dtype=np.uint8)
        _target = Image.fromarray(_target)

        # synchrosized transform
        if self.train:
            _img, _target = self._sync_transform( _img, _target)

        # general resize, normalize and toTensor
        if self.transform is not None:
            _img = self.transform(_img)
        if self.target_transform is not None:
            _target = self.target_transform(_target)

        return _img, _target

    def __len__(self):
        return len(self.images)

    def _sync_transform(self, img, mask):
        # random rotate -10~10
        deg = random.uniform(-10,10)
        img = img.rotate(deg)
        mask = mask.rotate(deg, PIL.Image.NEAREST)

        return img, mask

if __name__ == '__main__':
    hpw18 = Segmentation_HPW18('/cvdata1/lliuqian/', train=True)
    print(hpw18[0])
    print (len(hpw18))
