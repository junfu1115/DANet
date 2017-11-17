##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## ECE Department, Rutgers University
## Email: zhang.hang@rutgers.edu
## Copyright (c) 2017
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import os
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

import encoding.dilated as dresnet
import torchvision.models as orgresnet

class Dilated_ResNet(nn.Module):
    def __init__(self, nclass):
        super(Dilated_ResNet, self).__init__()
        self.pretrained = dresnet.resnet50(pretrained=True)

    def forward(self, x):
        # pre-trained ResNet feature
        x = self.pretrained.conv1(x)
        x = self.pretrained.bn1(x)
        x = self.pretrained.relu(x)
        x = self.pretrained.maxpool(x)

        x = self.pretrained.layer1(x)
        x = self.pretrained.layer2(x)
        x = self.pretrained.layer3(x)
        x = self.pretrained.layer4(x)
        return x
        

class Org_ResNet(nn.Module):
    def __init__(self, nclass):
        super(Org_ResNet, self).__init__()
        self.pretrained = orgresnet.resnet50(pretrained=True)

    def forward(self, x):
        # pre-trained ResNet feature
        x = self.pretrained.conv1(x)
        x = self.pretrained.bn1(x)
        x = self.pretrained.relu(x)
        x = self.pretrained.maxpool(x)

        x = self.pretrained.layer1(x)
        x = self.pretrained.layer2(x)
        x = self.pretrained.layer3(x)
        x = self.pretrained.layer4(x)
        return x
        
def test_resnet():   
    # test the model
    model1 = Dilated_ResNet(10).eval().cuda()
    model2 = Org_ResNet(10).eval().cuda()
    model1.eval()
    model2.eval()
    x = Variable(torch.Tensor(1,3, 224, 224).uniform_(-0.5,0.5)).cuda()
    y1 = model1(x)
    y2 = model2(x)
    print(y1[0][1])
    print(y2[0][1])

if __name__ == "__main__":
    test_resnet()
