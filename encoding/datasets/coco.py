import os
from tqdm import tqdm, trange
from PIL import Image, ImageOps, ImageFilter

import torch
import torch.utils.data as data
import torchvision.transforms as transform
import random
import math
import numpy as np

from .dataset import ToLabel

"""
NUM_CHANNEL = 91
[] background
[5] airplane
[2] bicycle
[16] bird
[9] boat
[44] bottle
[6] bus
[3] car
[17] cat
[62] chair
[21] cow
[67] dining table
[18] dog
[19] horse
[4] motorcycle
[1] person
[64] potted plant
[20] sheep
[63] couch
[7] train
[72] tv
"""
catlist = [0, 5, 2, 16, 9, 44, 6, 3, 17, 62, 21, 67, 18, 19, 4,
    1, 64, 20, 63, 7, 72]


class Segmentation(data.Dataset):
    def __init__(self, root, mode='train', transform=None, 
                 target_transform=None):
        from pycocotools.coco import COCO
        from pycocotools import mask
        if mode == 'train':
            print('train set')
            ann_file = os.path.join(root, 'coco/annotations/instances_train2014.json')
            ids_file = os.path.join(root, 'coco/annotations/train_ids.pth')
            root = os.path.join(root, 'coco/train2014')
        else:
            print('val set')
            ann_file = os.path.join(root, 'coco/annotations/instances_val2014.json')
            ids_file = os.path.join(root, 'coco/annotations/val_ids.pth')
            root = os.path.join(root, 'coco/val2014')
        self.train = mode
        self.root = root
        self.coco = COCO(ann_file)
        self.coco_mask = mask
        if os.path.exists(ids_file):
            self.ids = torch.load(ids_file)
        else:
            self.new_ids = []
            self.ids = list(self.coco.imgs.keys())
            self.preprocess(ids_file)
            self.ids = self.new_ids
        self.transform = transform
        self.target_transform = target_transform

    def preprocess(self, ids_file):
        tbar = trange(len(self.ids))
        for i in tbar:
            img_id = self.ids[i]
            cocotarget = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
            img_metadata = self.coco.loadImgs(img_id)[0]
            mask = self._gen_seg_mask(cocotarget, img_metadata['height'], 
                                      img_metadata['width'])
            # more than 1k pixels
            if (mask > 0).sum() > 1000:
                self.new_ids.append(img_id)
            tbar.set_description('Doing: {}/{}, got {} qualified images'.\
                format(i, len(self.ids), len(self.new_ids)))

        print('number of qualified images: ', len(self.new_ids))
        torch.save(self.new_ids, ids_file)

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        cocotarget = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
        img_metadata = coco.loadImgs(img_id)[0]
        path = img_metadata['file_name']

        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        mask = Image.fromarray(
            self._gen_seg_mask(cocotarget, img_metadata['height'], 
                               img_metadata['width'])
            )
        # synchrosized transform
        if True:#self.train == 'train':
            img, mask = self._sync_transform(img, mask)
        else:
            img, mask = self._val_sync_transform(img, mask)

        # general resize, normalize and toTensor
        if self.transform is not None:
            #print("transform for input")
            img = self.transform(img)
        if self.target_transform is not None:
            #print("transform for label")
            mask = self.target_transform(mask)

        return img, mask

    def __len__(self):
        return len(self.ids)

    def _val_sync_transform(self, img, mask):
        outsize = 480
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
        base_size = 520
        crop_size = 480
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

    def _gen_seg_mask(self, target, h, w):
        mask = np.zeros((h, w), dtype=np.uint8)
        coco_mask = self.coco_mask
        for instance in target:
            rle = coco_mask.frPyObjects(instance['segmentation'], h, w)
            m = coco_mask.decode(rle)
            cat = instance['category_id']
            if cat in catlist:
                c = catlist.index(cat)
            else:
                continue
            if len(m.shape) < 3:
                mask[:, :] += (mask == 0) * (m * c)
            else:
                mask[:, :] += (mask == 0) * (((np.sum(m, axis=2)) > 0) * c).astype(np.uint8)
        return mask
