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
import threading
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function, Variable
from torch.nn.modules.utils import _single, _pair, _triple

from .._ext import encoding_lib

__all__ = ['dilatedavgpool2d']

class _dilatedavgpool2d(Function):
    def forward(self, input, kernel_size, stride, padding,
            dilation=1):
        self.kH, self.kW = _pair(kernel_size)
        self.dH, self.dW = _pair(stride if stride is not None else 
            kernel_size)
        self.padH, self.padW = _pair(padding)
        self.dilationH, self.dilationW = _pair(dilation)
        b,c,h,w = input.size()
        if self.dH==1 and self.dW==1:
            # keep the size for dilated avgpool
            ow, oh = w, h
        else:
            ow = math.floor(float(w-self.kW+2*self.padW)/float(self.dW)) +1
            oh = math.floor(float(h-self.kH+2*self.padH)/float(self.dH)) +1
        output = input.new(b,c,oh,ow)
        self.save_for_backward(input)
        encoding_lib.Encoding_Float_DilatedAvgPool2d_Forward(input, output,
            self.kH, self.kW, self.dH, self.dW, self.padH, self.padW,
            self.dilationH, self.dilationW)
        return output

    def backward(self, gradOutput):
        input, = self.saved_variables
        gradInput = input.new().resize_as_(input)
        encoding_lib.Encoding_Float_DilatedAvgPool2d_Backward(
            gradinput, gradoutput,
            self.kH, self.kW, self.dH, self.dW, self.padH, self.padW,
            self.dilationH, self.dilationW)
        return gradInput, None, None, None, None


def dilatedavgpool2d(input, kernel_size, stride=None, padding=0, 
        dilation=1):
    """Dilated Average Pool 2d, for dilation of DenseNet. 

    Applies 2D average-pooling operation in kh x kw regions by step size
    dh x dw steps. The number of output features is equal to the number of
    input planes.

    See :class:`~encoding.nn.DilatedAvgPool2d` for details and output shape.

    Args:
        input: input tensor (minibatch x in_channels x iH x iW)
        kernel_size: size of the pooling region, a single number or a
          tuple (kh x kw)
        stride: stride of the pooling operation, a single number or a
          tuple (sh x sw). Default is equal to kernel size
        padding: implicit zero padding on the input, a single number or
          a tuple (padh x padw), Default: 0
        dilation: the dilation parameter similar to Conv2d
    """
    return _dilatedavgpool2d.apply(input, kernel_size, stride, padding,
            dilation)
