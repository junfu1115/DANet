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

import torch.utils.data as data
from PIL import Image, ImageOps

import torch.utils.data as data
import torchvision.transforms as transform
from .dataset import ToLabel

class FolderLoader(data.Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.images = get_folder_images(root)
        if len(self.images) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: \
                " + self.root + "\n"))

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, os.path.basename(self.images[index])

    def __len__(self):
        return len(self.images)


def get_folder_images(img_folder):
    img_paths = []  
    for filename in os.listdir(img_folder):
        if filename.endswith(".jpg"):
            imgpath = os.path.join(img_folder, filename)
            img_paths.append(imgpath)
    return img_paths



class Dataloder():
    def __init__(self, args):
        # the data augmentation is implemented as part of the dataloader
        assert(args.test)
        input_transform = transform.Compose([
            transform.ToTensor(),
            transform.Normalize(args.mean, args.std)])
        args.test_batch_size = 1 

        assert(args.test_folder is not None)
        print('loading the data from: {}'.format(args.test_folder))

        testset = FolderLoader(args.test_folder, input_transform)
        kwargs = {'num_workers': args.workers, 'pin_memory': True} \
            if args.cuda else {}
        self.trainloader = None
        self.testloader = data.DataLoader(testset,
                                     batch_size=args.test_batch_size,
                                     shuffle=False, **kwargs)

    def getloader(self):
        return self.trainloader, self.testloader
