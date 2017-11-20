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
from torch.autograd import Variable
from torch.nn import Module, Parameter
from torch.nn import functional as F

from ..parallel import my_data_parallel
from .syncbn import BatchNorm2d
from ..functions import view_each, upsample
from .basic import *

__all__ = ['GramMatrix', 'View', 'Sum', 'Mean', 'Normalize', 'PyramidPooling']


class GramMatrix(Module):
    r""" Gram Matrix for a 4D convolutional featuremaps as a mini-batch

    .. math::
        \mathcal{G} = \sum_{h=1}^{H_i}\sum_{w=1}^{W_i} \mathcal{F}_{h,w}\mathcal{F}_{h,w}^T
    """
    def forward(self, y):
        (b, ch, h, w) = y.size()
        features = y.view(b, ch, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (ch * h * w)
        return gram


class View(Module):
    """Reshape the input into different size, an inplace operator, support
    SelfParallel mode.
    """
    def __init__(self, *args):
        super(View, self).__init__()
        if len(args) == 1 and isinstance(args[0], torch.Size):
            self.size = args[0]
        else:
            self.size = torch.Size(args)

    def forward(self, input):
        if isinstance(input, Variable):
            return input.view(self.size)
        elif isinstance(input, tuple) or isinstance(input, list):
            return view_each(input, self.size)
        else:
            raise RuntimeError('unknown input type')


class Sum(Module):
    def __init__(self, dim, keep_dim=False):
        super(Sum, self).__init__()
        self.dim = dim
        self.keep_dim = keep_dim

    def forward(self, input):
        if isinstance(input, Variable):
            return input.sum(self.dim, self.keep_dim)
        elif isinstance(input, tuple) or isinstance(input, list):
            return my_data_parallel(self, input)
        else:
            raise RuntimeError('unknown input type')


class Mean(Module):
    def __init__(self, dim, keep_dim=False):
        super(Mean, self).__init__()
        self.dim = dim
        self.keep_dim = keep_dim

    def forward(self, input):
        if isinstance(input, Variable):
            return input.mean(self.dim, self.keep_dim)
        elif isinstance(input, tuple) or isinstance(input, list):
            return my_data_parallel(self, input)
        else:
            raise RuntimeError('unknown input type')


class Normalize(Module):
    r"""Performs :math:`L_p` normalization of inputs over specified dimension.

    Does:

    .. math::
        v = \frac{v}{\max(\lVert v \rVert_p, \epsilon)}

    for each subtensor v over dimension dim of input. Each subtensor is
    flattened into a vector, i.e. :math:`\lVert v \rVert_p` is not a matrix
    norm.

    With default arguments normalizes over the second dimension with Euclidean
    norm.

    Args:
        p (float): the exponent value in the norm formulation. Default: 2
        dim (int): the dimension to reduce. Default: 1
    """
    def __init__(self, p=2, dim=1):
        super(Normalize, self).__init__()
        self.p = p
        self.dim =dim

    def forward(self, x):
        if isinstance(x, Variable):
            return F.normalize(x, self.p, self.dim, eps=1e-10)
        elif isinstance(x, tuple) or isinstance(x, list):
            return my_data_parallel(self, x)
        else:
            raise RuntimeError('unknown input type')


class PyramidPooling(Module):
    """
    Reference: 
        Zhao, Hengshuang, et al. *"Pyramid scene parsing network."*
    """
    def __init__(self, in_channels):
        super(PyramidPooling, self).__init__()
        self.pool1 = AdaptiveAvgPool2d(1)
        self.pool2 = AdaptiveAvgPool2d(2)
        self.pool3 = AdaptiveAvgPool2d(3)
        self.pool4 = AdaptiveAvgPool2d(6)

        out_channels = int(in_channels/4)
        self.conv1 = Sequential(Conv2d(in_channels, out_channels, 1),
                                BatchNorm2d(out_channels),
                                ReLU(True))
        self.conv2 = Sequential(Conv2d(in_channels, out_channels, 1),
                                BatchNorm2d(out_channels),
                                ReLU(True))
        self.conv3 = Sequential(Conv2d(in_channels, out_channels, 1),
                                BatchNorm2d(out_channels),
                                ReLU(True))
        self.conv4 = Sequential(Conv2d(in_channels, out_channels, 1),
                                BatchNorm2d(out_channels),
                                ReLU(True))

    def _cat_each(self, x, feat1, feat2, feat3, feat4):
        assert(len(x)==len(feat1))
        z = []
        for i in range(len(x)):
            z.append( torch.cat((x[i], feat1[i], feat2[i], feat3[i], feat4[i]), 1))
        return z

    def forward(self, x):
        if isinstance(x, Variable):
            _, _, h, w = x.size()
        elif isinstance(x, tuple) or isinstance(x, list):
            _, _, h, w = x[0].size()
        else:
            raise RuntimeError('unknown input type')
        feat1 = upsample(self.conv1(self.pool1(x)),(h,w),
                              mode='bilinear')
        feat2 = upsample(self.conv2(self.pool2(x)),(h,w),
                              mode='bilinear')
        feat3 = upsample(self.conv3(self.pool3(x)),(h,w), 
                              mode='bilinear')
        feat4 = upsample(self.conv4(self.pool4(x)),(h,w), 
                              mode='bilinear')
        if isinstance(x, Variable):
            return torch.cat((x, feat1, feat2, feat3, feat4), 1)
        elif isinstance(x, tuple) or isinstance(x, list):
            return self._cat_each(x, feat1, feat2, feat3, feat4)
        else:
            raise RuntimeError('unknown input type')

