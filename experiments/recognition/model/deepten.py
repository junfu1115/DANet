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
from torch.autograd import Variable
import torch.nn as nn
from torch.autograd import Variable

import encoding
import encoding.dilated.resnet as resnet

class Net(nn.Module):
    def __init__(self, args):
        nclass=args.nclass
        super(Net, self).__init__()
        self.backbone = args.backbone
        # copying modules from pretrained models
        if self.backbone == 'resnet50':
            self.pretrained = resnet.resnet50(pretrained=True, dilated=False)
        elif self.backbone == 'resnet101':
            self.pretrained = resnet.resnet101(pretrained=True, dilated=False)
        elif self.backbone == 'resnet152':
            self.pretrained = resnet.resnet152(pretrained=True, dilated=False)
        else:
            raise RuntimeError('unknown backbone: {}'.format(self.backbone))
        n_codes = 32
        self.head = nn.Sequential(
            nn.Conv2d(2048, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            encoding.nn.Encoding(D=128,K=n_codes),
            encoding.nn.View(-1, 128*n_codes),
            encoding.nn.Normalize(),
            nn.Linear(128*n_codes, nclass),
        )

    def forward(self, x):
        if isinstance(x, Variable):
            _, _, h, w = x.size()
        elif isinstance(x, tuple) or isinstance(x, list):
            var_input = x 
            while not isinstance(var_input, Variable):
                var_input = var_input[0]
            _, _, h, w = var_input.size()
        else:
            raise RuntimeError('unknown input type: ', type(x))
        x = self.pretrained.conv1(x)
        x = self.pretrained.bn1(x)
        x = self.pretrained.relu(x)
        x = self.pretrained.maxpool(x)
        x = self.pretrained.layer1(x)
        x = self.pretrained.layer2(x)
        x = self.pretrained.layer3(x)
        x = self.pretrained.layer4(x)
        return self.head(x)

