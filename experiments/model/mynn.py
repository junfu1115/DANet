##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## ECE Department, Rutgers University
## Email: zhang.hang@rutgers.edu
## Copyright (c) 2017
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import encoding
from encoding import Encoding

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
                     nn.Conv2d(inplanes, planes, kernel_size=3,
                         stride=stride, padding=1),
                     norm_layer(planes),
                     nn.ReLU(inplace=True),
                     nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                         padding=1)]
        self.conv_block = nn.Sequential(*conv_block)
    
    def forward(self, input):
        #print(input.size())
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
                           stride=1)]
        conv_block += [norm_layer(planes),
                       nn.ReLU(inplace=True),
                       nn.Conv2d(planes, planes, kernel_size=3, 
                           stride=stride, padding=1)]
        conv_block += [norm_layer(planes),
                       nn.ReLU(inplace=True),
                       nn.Conv2d(planes, planes * self.expansion, 
                           kernel_size=1, stride=1)]
        self.conv_block = nn.Sequential(*conv_block)
        
    def forward(self, x):
        if self.downsample:
            residual = self.residual_layer(x)
        else:
            residual = x
        return residual + self.conv_block(x)
        

##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class ResNeXtBlock(nn.Module):
    """
    Aggregated Residual Transformations for Deep Neural Networks
    ref https://arxiv.org/abs/1611.05431
    """
    def __init__(self, inplanes, planes, cardinality=32, base_width=4, 
            stride=1, expansion=4):
        super(ResNeXtBlock, self).__init__()
        width = int(math.floor(planes * (base_width/64.0)))
        group_width = cardinality * width
        conv_block = []
        conv_block += [
            nn.Conv2d(inplanes, group_width, kernel_size=1, bias=False),
            nn.BatchNorm2d(group_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(group_width, group_width, kernel_size=3, 
                stride=stride, padding=1, groups=cardinality, bias=False),
            nn.BatchNorm2d(group_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(group_width, expansion*group_width, kernel_size=1, 
                bias=False),
            nn.BatchNorm2d(expansion*group_width),
            nn.ReLU(inplace=True)]
        self.conv_block = nn.Sequential(*conv_block)
        if stride != 1 or inplanes != expansion*group_width:
            self.downsample = True
            self.residual_layer = nn.Conv2d(inplanes, 
                expansion*group_width, kernel_size=1, stride=stride, 
                bias=False)
        else:
            self.downsample = False

    def forward(self, x):
        if self.downsample:
            residual = self.residual_layer(x)
        else:
            residual = x
        return residual + self.conv_block(x)


class View(nn.Module):
    def __init__(self, *args):
        super(View, self).__init__()
        if len(args) == 1 and isinstance(args[0], torch.Size):
            self.size = args[0]
        else:
            self.size = torch.Size(args)

    def forward(self, input):
        return input.view(self.size)


##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class DenseBlock(nn.Module):
    """
    Densely Connected Convolutional Networks
    ref https://arxiv.org/abs/1608.06993
    """
    def __init__(self, in_planes, growth_rate):
        super(DenseBlock, self).__init__()
        model = []
        model += [nn.BatchNorm2d(in_planes),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(in_planes, 4*growth_rate, kernel_size=1, 
                     bias=False),
                  nn.BatchNorm2d(4*growth_rate),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(4*growth_rate, growth_rate, kernel_size=3, 
                     padding=1, bias=False)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        out = self.model(x)
        out = torch.cat([out,x], 1)
        return out


class Transition(nn.Module):
    """
    Densely Connected Convolutional Networks
    ref https://arxiv.org/abs/1608.06993
    """
    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()
        model = []
        model += [nn.BatchNorm2d(in_planes),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(in_planes, out_planes, kernel_size=1,
                      bias=False),
                  nn.AvgPool2d(2)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        out_channel = int(channel / reduction)
        self.fc = nn.Sequential(
            nn.Linear(channel, out_channel),
            nn.ReLU(inplace=True),
            nn.Linear(out_channel, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, reduction=16):
        super(SEBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SELayer(planes, reduction)
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


class SEBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(planes * 4, reduction)
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
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)

        if self.downsample:
            residual = self.residual_layer(x)

        out += residual
        out = self.relu(out)

        return out


##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class ELayer(nn.Module):
    def __init__(self, channel, K=16, reduction=4):
        super(ELayer, self).__init__()
        out_channel = int(channel / reduction)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, out_channel, 1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            Encoding(D=out_channel,K=K),
            nn.BatchNorm1d(K),
            nn.ReLU(inplace=True),
            View(-1, out_channel*K),
            nn.Linear(out_channel*K, channel),
            nn.Sigmoid()
        )
        """
        encoding.nn.View(-1, out_channel*K),
        encoding.Normalize(),
        """

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.fc(x).view(b, c, 1, 1)
        return x * y

class EBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, K=16):
        super(EBasicBlock, self).__init__()
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


class EBottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None, K=16):
        super(EBottleneck, self).__init__()
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



class EResNeXtBottleneck(nn.Module):
  expansion = 4
  """
  RexNeXt bottleneck type C (https://github.com/facebookresearch/ResNeXt/blob/master/models/resnext.lua)
  """
  def __init__(self, inplanes, planes, cardinality, base_width, stride=1, downsample=None, K=32):
    super(EResNeXtBottleneck, self).__init__()

    D = int(math.floor(planes * (base_width/64.0)))
    C = cardinality

    self.conv_reduce = nn.Conv2d(inplanes, D*C, kernel_size=1, stride=1, padding=0, bias=False)
    self.bn_reduce = nn.BatchNorm2d(D*C)

    self.conv_conv = nn.Conv2d(D*C, D*C, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
    self.bn = nn.BatchNorm2d(D*C)

    self.conv_expand = nn.Conv2d(D*C, planes*4, kernel_size=1, stride=1, padding=0, bias=False)
    self.bn_expand = nn.BatchNorm2d(planes*4)
    self.se = ELayer(planes * 4, K, self.expansion*4)

    self.downsample = downsample

  def forward(self, x):
    residual = x

    bottleneck = self.conv_reduce(x)
    bottleneck = F.relu(self.bn_reduce(bottleneck), inplace=True)

    bottleneck = self.conv_conv(bottleneck)
    bottleneck = F.relu(self.bn(bottleneck), inplace=True)

    bottleneck = self.conv_expand(bottleneck)
    bottleneck = self.bn_expand(bottleneck)

    bottleneck = self.se(bottleneck)

    if self.downsample is not None:
      residual = self.downsample(x)
    
    return F.relu(residual + bottleneck, inplace=True)
