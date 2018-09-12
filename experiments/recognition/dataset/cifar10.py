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
import torchvision
import torchvision.transforms as transforms

class Dataloader():
    def __init__(self, args):
        transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                (0.2023, 0.1994, 0.2010)),
        ])

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, 
            download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, 
            download=True, transform=transform_test)
    
        kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=
            args.batch_size, shuffle=True, **kwargs)
        testloader = torch.utils.data.DataLoader(testset, batch_size=
            args.batch_size, shuffle=False, **kwargs)
        self.trainloader = trainloader 
        self.testloader = testloader
    
    def getloader(self):
        return self.trainloader, self.testloader
