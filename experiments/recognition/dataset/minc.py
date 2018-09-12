##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## ECE Department, Rutgers University
## Email: zhang.hang@rutgers.edu
## Copyright (c) 2017
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import torch
import torch.utils.data as data
import torchvision
from torchvision import transforms

from PIL import Image
import os
import os.path

_imagenet_pca = {
    'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
    'eigvec': torch.Tensor([
        [-0.5675,  0.7192,  0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948,  0.4203],
    ])
}


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(filename, datadir, class_to_idx):
    images = []
    labels = []
    with open(os.path.join(filename), "r") as lines:
        for line in lines:
            _image = os.path.join(datadir, line.rstrip('\n'))
            _dirname = os.path.split(os.path.dirname(_image))[1]
            assert os.path.isfile(_image)
            label = class_to_idx[_dirname]
            images.append(_image)
            labels.append(label)

    return images, labels


class MINCDataloder(data.Dataset):
    def __init__(self, root, train=True, transform=None):
        self.transform = transform
        classes, class_to_idx = find_classes(root + '/images')
        if train:
            filename = os.path.join(root, 'labels/train1.txt')
        else:
            filename = os.path.join(root, 'labels/test1.txt')

        self.images, self.labels = make_dataset(filename, root, 
            class_to_idx)
        assert (len(self.images) == len(self.labels))

    def __getitem__(self, index):
        _img = Image.open(self.images[index]).convert('RGB')
        _label = self.labels[index]
        if self.transform is not None:
            _img = self.transform(_img)

        return _img, _label

    def __len__(self):
        return len(self.images)


class Dataloader():
    def __init__(self, args):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4,0.4,0.4),
            transforms.ToTensor(),
            Lighting(0.1, _imagenet_pca['eigval'], _imagenet_pca['eigvec']),
            normalize,
        ])
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

        trainset = MINCDataloder(root=os.path.expanduser('~/data/minc-2500/'), 
            train=True, transform=transform_train)
        testset = MINCDataloder(root=os.path.expanduser('~/data/minc-2500/'), 
            train=False, transform=transform_test)
    
        kwargs = {'num_workers': 8, 'pin_memory': True} if args.cuda else {}
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=
            args.batch_size, shuffle=True, **kwargs)
        testloader = torch.utils.data.DataLoader(testset, batch_size=
            args.test_batch_size, shuffle=False, **kwargs)
        self.trainloader = trainloader 
        self.testloader = testloader
    
    def getloader(self):
        return self.trainloader, self.testloader


class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone()\
            .mul(alpha.view(1, 3).expand(3, 3))\
            .mul(self.eigval.view(1, 3).expand(3, 3))\
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))


if __name__ == "__main__":
    trainset = MINCDataloder(root=os.path.expanduser('~/data/minc-2500/'), train=True)
    testset = MINCDataloder(root=os.path.expanduser('~/data/minc-2500/'), train=False)
    print(len(trainset))
    print(len(testset))
