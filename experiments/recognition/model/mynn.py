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
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import encoding

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class Basicblock(nn.Module):
    """ Pre-activation residual block
    Identity Mapping in Deep Residual Networks
    ref https://arxiv.org/abs/1603.05027
    """
    def __init__(self, inplanes, planes, stride=1, 
                            norm_layer=nn.BatchNorm2d):
        super(Basicblock, self).__init__()
        if inplanes != planes or stride !=1 :
            self.downsample = True
            self.residual_layer = nn.Conv2d(inplanes, planes,
                                            kernel_size=1, stride=stride)
        else:
            self.downsample = False
        conv_block=[]
        conv_block+=[norm_layer(inplanes),
                     nn.ReLU(inplace=True),
                     conv3x3(inplanes, planes,stride=stride),
                     norm_layer(planes),
                     nn.ReLU(inplace=True),
                     conv3x3(planes, planes)]
        self.conv_block = nn.Sequential(*conv_block)
    
    def forward(self, input):
        if self.downsample:
            residual = self.residual_layer(input)
        else:
            residual = input
        return residual + self.conv_block(input)


class Bottleneck(nn.Module):
    """ Pre-activation residual block
    Identity Mapping in Deep Residual Networks
    ref https://arxiv.org/abs/1603.05027
    """
    def __init__(self, inplanes, planes, stride=1,norm_layer=nn.BatchNorm2d):
        super(Bottleneck, self).__init__()
        self.expansion = 4
        if inplanes != planes*self.expansion or stride !=1 :
            self.downsample = True
            self.residual_layer = nn.Conv2d(inplanes, 
                planes * self.expansion, kernel_size=1, stride=stride)
        else:
            self.downsample = False
        conv_block = []
        conv_block += [norm_layer(inplanes),
                       nn.ReLU(inplace=True),
                       nn.Conv2d(inplanes, planes, kernel_size=1,
                                 stride=1, bias=False)]
        conv_block += [norm_layer(planes),
                       nn.ReLU(inplace=True),
                       nn.Conv2d(planes, planes, kernel_size=3, 
                                 stride=stride, padding=1, bias=False)]
        conv_block += [norm_layer(planes),
                       nn.ReLU(inplace=True),
                       nn.Conv2d(planes, planes * self.expansion, 
                                 kernel_size=1, stride=1, bias=False)]
        self.conv_block = nn.Sequential(*conv_block)
        
    def forward(self, x):
        if self.downsample:
            residual = self.residual_layer(x)
        else:
            residual = x
        return residual + self.conv_block(x)
        
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class EncLayer(nn.Module):
    def __init__(self, channel, K=16, reduction=4):
        super(EncLayer, self).__init__()
        out_channel = int(channel / reduction)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, out_channel, 1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            encoding.nn.Encoding(D=out_channel,K=K),
            encoding.nn.View(-1, out_channel*K),
            encoding.nn.Normalize(),
            nn.Linear(out_channel*K, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.fc(x).view(b, c, 1, 1)
        return x * y


class EncDropLayer(nn.Module):
    def __init__(self, channel, K=16, reduction=4):
        super(EncDropLayer, self).__init__()
        out_channel = int(channel / reduction)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, out_channel, 1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            encoding.nn.EncodingDrop(D=out_channel,K=K),
            encoding.nn.View(-1, out_channel*K),
            encoding.nn.Normalize(),
            nn.Linear(out_channel*K, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.fc(x).view(b, c, 1, 1)
        return x * y


class EncBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, K=16, ELayer=EncLayer):
        super(EncBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = ELayer(planes, K, self.expansion*4)
        self.stride = stride
        if inplanes != planes or stride !=1 :
            self.downsample = True
            self.residual_layer = nn.Conv2d(inplanes, planes,
                                            kernel_size=1, stride=stride)
        else:
            self.downsample = False

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample:
            residual = self.residual_layer(x)

        out += residual
        out = self.relu(out)

        return out


class EncBottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None, K=16, ELayer=EncLayer):
        super(EncBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 
            kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.se = ELayer(planes * self.expansion, K, self.expansion*4)
        self.stride = stride
        if inplanes != planes * self.expansion or stride !=1 :
            self.downsample = True
            self.residual_layer = nn.Conv2d(inplanes, 
                planes* self.expansion, kernel_size=1, stride=stride)
        else:
            self.downsample = False

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)

        if self.downsample:
            residual = self.residual_layer(x)
        out += residual
        out = self.relu(out)

        return out
