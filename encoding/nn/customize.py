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
from torch.nn import Module
from torch.nn import functional as F
from torch.nn.parameter import Parameter

from ..parallel import my_data_parallel
from .syncbn import BatchNorm2d
from ..functions import dilatedavgpool2d, view_each, upsample
from .basic import *

__all__ = ['DilatedAvgPool2d', 'UpsampleConv2d', 'View', 'Sum', 'Mean', 
    'Normalize', 'Bottleneck', 'PyramidPooling']

class DilatedAvgPool2d(Module):
    r"""We provide Dilated Average Pooling for the dilation of Densenet as
    in :class:`encoding.dilated.DenseNet`.

    Reference::
        We provide this code for a comming paper.

    Applies a 2D average pooling over an input signal composed of several input planes.

    In the simplest case, the output value of the layer with input size :math:`(N, C, H, W)`,
    output :math:`(N, C, H_{out}, W_{out})` and :attr:`kernel_size` :math:`(kH, kW)`
    can be precisely described as:

    .. math::

        \begin{array}{ll}
        out(b, c, h, w)  = 1 / (kH * kW) * 
        \sum_{{m}=0}^{kH-1} \sum_{{n}=0}^{kW-1}
        input(b, c, dH * h + m, dW * w + n)
        \end{array}

    | If :attr:`padding` is non-zero, then the input is implicitly zero-padded on both sides
      for :attr:`padding` number of points

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`dilation` can either be:

        - a single ``int`` -- in which case the same value is used for the height and width dimension
        - a ``tuple`` of two ints -- in which case, the first `int` is used for the height dimension,
          and the second `int` for the width dimension

    Args:
        kernel_size: the size of the window
        stride: the stride of the window. Default value is :attr:`kernel_size`
        padding: implicit zero padding to be added on both sides
        dilation: the dilation parameter similar to Conv2d

    Shape:
        - Input: :math:`(N, C, H_{in}, W_{in})`
        - Output: :math:`(N, C, H_{out}, W_{out})` where
          :math:`H_{out} = floor((H_{in}  + 2 * padding[0] - kernel\_size[0]) / stride[0] + 1)`
          :math:`W_{out} = floor((W_{in}  + 2 * padding[1] - kernel\_size[1]) / stride[1] + 1)`

    Examples::

        >>> # pool of square window of size=3, stride=2, dilation=2
        >>> m = nn.DilatedAvgPool2d(3, stride=2, dilation=2)
        >>> input = autograd.Variable(torch.randn(20, 16, 50, 32))
        >>> output = m(input)

    """
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1):
        super(DilatedAvgPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding
        self.dilation = dilation

    def forward(self, input):
        if isinstance(input, Variable):
            return dilatedavgpool2d(input, self.kernel_size, self.stride,
                                self.padding, self.dilation)
        elif isinstance(input, tuple) or isinstance(input, list):
            return my_data_parallel(self, input)
        else:
            raise RuntimeError('unknown input type')

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + 'size=' + str(self.kernel_size) \
            + ', stride=' + str(self.stride) \
            + ', padding=' + str(self.padding) \
            + ', dilation=' + str(self.dilation) + ')'


class UpsampleConv2d(Module):
    r"""
    To avoid the checkerboard artifacts of standard Fractionally-strided Convolution, we adapt an integer stride convolution but producing a :math:`2\times 2` outputs for each convolutional window. 

    .. image:: _static/img/upconv.png
        :width: 50%
        :align: center

    Reference:
        Hang Zhang and Kristin Dana. "Multi-style Generative Network for Real-time Transfer."  *arXiv preprint arXiv:1703.06953 (2017)*

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        output_padding (int or tuple, optional): Zero-padding added to one side of the output. Default: 0
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        bias (bool, optional): If True, adds a learnable bias to the output. Default: True
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        scale_factor (int): scaling factor for upsampling convolution. Default: 1

    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})` where
          :math:`H_{out} = scale * (H_{in} - 1) * stride[0] - 2 * padding[0] + kernel\_size[0] + output\_padding[0]`
          :math:`W_{out} = scale * (W_{in} - 1) * stride[1] - 2 * padding[1] + kernel\_size[1] + output\_padding[1]`

    Attributes:
        weight (Tensor): the learnable weights of the module of shape
                         (in_channels, scale * scale * out_channels, kernel_size[0], kernel_size[1])
        bias (Tensor):   the learnable bias of the module of shape (scale * scale * out_channels)

    Examples::
        >>> # With square kernels and equal stride
        >>> m = nn.UpsampleCov2d(16, 33, 3, stride=2)
        >>> # non-square kernels and unequal stride and with padding
        >>> m = nn.UpsampleCov2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
        >>> input = autograd.Variable(torch.randn(20, 16, 50, 100))
        >>> output = m(input)
        >>> # exact output size can be also specified as an argument
        >>> input = autograd.Variable(torch.randn(1, 16, 12, 12))
        >>> downsample = nn.Conv2d(16, 16, 3, stride=2, padding=1)
        >>> upsample = nn.UpsampleCov2d(16, 16, 3, stride=2, padding=1)
        >>> h = downsample(input)
        >>> h.size()
        torch.Size([1, 16, 6, 6])
        >>> output = upsample(h, output_size=input.size())
        >>> output.size()
        torch.Size([1, 16, 12, 12])

    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, scale_factor =1, 
                 bias=True):
        super(UpsampleConv2d, self).__init__()
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.scale_factor = scale_factor
        self.weight = Parameter(torch.Tensor(
            out_channels * scale_factor * scale_factor, 
            in_channels // groups, *kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels * 
                scale_factor * scale_factor))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        if isinstance(input, Variable):
            out = F.conv2d(input, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
            return F.pixel_shuffle(out, self.scale_factor)
        elif isinstance(input, tuple) or isinstance(input, list):
            return my_data_parallel(self, input)
        else:
            raise RuntimeError('unknown input type')


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


class Bottleneck(Module):
    """ Pre-activation residual block
    Identity Mapping in Deep Residual Networks
    ref https://arxiv.org/abs/1603.05027
    """
    def __init__(self, inplanes, planes, stride=1,
            norm_layer=BatchNorm2d):
        super(Bottleneck, self).__init__()
        self.expansion = 4
        if inplanes != planes*self.expansion or stride !=1 :
            self.downsample = True
            self.residual_layer = Conv2d(inplanes, planes * self.expansion,
                                                        kernel_size=1, stride=stride)
        else:
            self.downsample = False
        conv_block = []
        conv_block += [norm_layer(inplanes),
                       ReLU(inplace=True),
                       Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)]
        conv_block += [norm_layer(planes),
                       ReLU(inplace=True),
                       Conv2d(planes, planes, kernel_size=3, stride=stride,
                           padding=1, bias=False)]
        conv_block += [norm_layer(planes),
                       ReLU(inplace=True),
                       Conv2d(planes, planes * self.expansion, kernel_size=1,
                           stride=1, bias=False)]
        self.conv_block = Sequential(*conv_block)
        
    def forward(self, x):
        if self.downsample:
            residual = self.residual_layer(x)
        else:
            residual = x
        if isinstance(x, Variable):
            return residual + self.conv_block(x)
        elif isinstance(x, tuple) or isinstance(x, list):
            return sum_each(residual, self.conv_block(x))
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

