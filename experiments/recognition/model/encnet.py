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
from .mynn import EncBasicBlock
import encoding

class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        num_blocks=[2,2,2]
        block=EncBasicBlock
        if block == EncBasicBlock:
            self.expansion = 1
        else:
            self.expansion = 4

        self.inplanes = args.widen * 16
        strides = [1, 2, 2]
        model = []
        # Conv_1
        model += [nn.Conv2d(3, self.inplanes, kernel_size=3, padding=1),
                  nn.BatchNorm2d(self.inplanes),
                  nn.ReLU(inplace=True)]
        # Residual units
        model += [self._residual_unit(block, self.inplanes, num_blocks[0],
                                      strides[0], args.ncodes)]
        for i in range(2):
            model += [self._residual_unit(block, 
                      int(2*self.inplanes/self.expansion), 
                      num_blocks[i+1], strides[i+1], args.ncodes)]
        # Last conv layer
        model += [nn.BatchNorm2d(self.inplanes),
                  nn.ReLU(inplace=True),
                  nn.AvgPool2d(8),
                  encoding.nn.View(-1, self.inplanes),
                  nn.Linear(self.inplanes, args.nclass)]

        self.model = nn.Sequential(*model)

    def _residual_unit(self, block, planes, n_blocks, stride, ncodes):
        strides = [stride] + [1]*(n_blocks-1)
        layers = []
        for i in range(n_blocks):
            layers += [block(self.inplanes, planes, strides[i], ncodes)]
            self.inplanes = self.expansion*planes
        return nn.Sequential(*layers)

    def forward(self, input):
        return self.model(input)
