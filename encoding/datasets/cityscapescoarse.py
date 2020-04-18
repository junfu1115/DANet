###########################################################################
# Created by: Hang Zhang
# Email: zhang.hang@rutgers.edu
# Copyright (c) 2017
###########################################################################

import os
import sys
import numpy as np
import random
import math
from tqdm import tqdm
from PIL import Image, ImageOps, ImageFilter

import torch
import torch.utils.data as data
import torchvision.transforms as transform

class Segmentation(data.Dataset):
    BASE_DIR = 'cityscapes'
    
    def __init__(self, data_folder, mode='train', transform=None, 
                 target_transform=None):
        self.root = os.path.join(data_folder, self.BASE_DIR)
        self.transform = transform
        self.target_transform = target_transform
        self.mode = mode
        self.images, self.masks = get_city_pairs(self.root, mode)
        assert (len(self.images) == len(self.masks))
        if len(self.images) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: \
                " + self.root + "\n"))

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        if self.mode == 'test':
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.images[index])

        mask = Image.open(self.masks[index])#.convert("P")
        mask = np.array(mask) 
        mask += 1
        mask[mask==256] = 0
        mask = Image.fromarray(mask)
        # synchrosized transform
        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'val':
            img, mask = self._val_sync_transform(img, mask)
        else:
            raise RuntimeError('unknown mode for dataloader: {}'.format(self.mode))
        

        # general resize, normalize and toTensor
        if self.transform is not None:
            #print("transform for input")
            img = self.transform(img)
        if self.target_transform is not None:
            #print("transform for label")
            mask = self.target_transform(mask)

        return img, mask

    def __len__(self):
        return len(self.images)

    def _val_sync_transform(self, img, mask):
        """
        synchronized transformation
        """
        outsize = 720
        short = outsize
        w, h = img.size
        if w > h:
            oh = short
            ow = int(1.0 * w * oh / h)
        else:
            ow = short
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # center crop
        w, h = img.size
        x1 = int(round((w - outsize) / 2.))
        y1 = int(round((h - outsize) / 2.))
        img = img.crop((x1, y1, x1+outsize, y1+outsize))
        mask = mask.crop((x1, y1, x1+outsize, y1+outsize))

        return img, mask

    def _sync_transform(self, img, mask):
        # random mirror
        if random.random() < 0.5:
            img  = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        base_size = 2048
        crop_size = 720
        # random scale (short edge from 480 to 720)
        long_size = random.randint(int(base_size*0.5), int(base_size*2.0))
        w, h = img.size
        if h > w:
            oh = long_size
            ow = int(1.0 * w * oh / h)
            short_size = ow
        else:
            ow = long_size
            oh = int(1.0 * h * ow / w)
            short_size = oh
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # random rotate -10~10, mask using NN rotate
        deg = random.uniform(-10,10)
        img = img.rotate(deg, resample=Image.BILINEAR)
        mask = mask.rotate(deg, resample=Image.NEAREST)
        # pad crop
        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            img  = ImageOps.expand(img,  border=(0,0,padw,padh), fill=0)
            mask = ImageOps.expand(mask, border=(0,0,padw,padh), fill=0)
        # random crop 480
        w, h = img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size) 
        img = img.crop((x1, y1, x1+crop_size, y1+crop_size))
        mask = mask.crop((x1, y1, x1+crop_size, y1+crop_size))
        # gaussian blur as in PSP ?
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))
        return img, mask


def get_city_pairs(folder, mode='train'):
    img_paths = []  
    mask_paths = []  
    if mode=='train':
        img_folder = os.path.join(folder, 'leftImg8bit/train_extra')
        mask_folder = os.path.join(folder, 'gtCoarse/train_extra')
    else:
        img_folder = os.path.join(folder, 'leftImg8bit/val')
        mask_folder = os.path.join(folder, 'gtFine/val')
    for root, directories, files in os.walk(img_folder):
        for filename in files:
            basename, extension =os.path.splitext(filename)
            if filename.endswith(".png"):
                imgpath = os.path.join(root, filename)
                foldername = os.path.basename(os.path.dirname(imgpath))
                maskname = filename.replace('leftImg8bit','gtCoarse_trainIds')
                maskpath = os.path.join(mask_folder, foldername, maskname)
                if os.path.isfile(imgpath) and os.path.isfile(maskpath):
                    img_paths.append(imgpath)
                    mask_paths.append(maskpath)
                else:
                    print('cannot find the mask or image:', imgpath, maskpath)

    return img_paths, mask_paths
