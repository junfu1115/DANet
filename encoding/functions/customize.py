##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## ECE Department, Rutgers University
## Email: zhang.hang@rutgers.edu
## Copyright (c) 2017
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

"""Encoding Customized Functions"""
import math
import torch
from torch.autograd import Function, Variable
from torch.nn.modules.utils import _pair

from .._ext import encoding_lib

__all__ = ['dilatedavgpool2d']

class _dilatedavgpool2d(Function):
    @staticmethod
    def forward(ctx, input, kernel_size, stride, padding,
                dilation=1):
        ctx.kH, ctx.kW = _pair(kernel_size)
        ctx.dH, ctx.dW = _pair(stride if stride is not None else kernel_size)
        ctx.padH, ctx.padW = _pair(padding)
        ctx.dilationH, ctx.dilationW = _pair(dilation)
        b, c, h, w = input.size()
        if ctx.dH == 1 and ctx.dW == 1:
            # keep the size for dilated avgpool
            ow, oh = w, h
        else:
            ow = math.floor(float(w-ctx.kW+2*ctx.padW)/float(ctx.dW)) +1
            oh = math.floor(float(h-ctx.kH+2*ctx.padH)/float(ctx.dH)) +1
        with torch.cuda.device_of(input):
            output = input.new(b, c, oh, ow)
        ctx.save_for_backward(input)
        if isinstance(input, torch.cuda.FloatTensor):
            with torch.cuda.device_of(input):
                encoding_lib.Encoding_Float_DilatedAvgPool2d_Forward(
                    input, output, ctx.kH, ctx.kW, ctx.dH, ctx.dW, ctx.padH,
                    ctx.padW, ctx.dilationH, ctx.dilationW)
        elif isinstance(input, torch.cuda.DoubleTensor):
            with torch.cuda.device_of(input):
                encoding_lib.Encoding_Double_DilatedAvgPool2d_Forward(
                    input, output, ctx.kH, ctx.kW, ctx.dH, ctx.dW, ctx.padH,
                    ctx.padW, ctx.dilationH, ctx.dilationW)
        else:
            raise RuntimeError('Unimplemented data type!')
        return output

    @staticmethod
    def backward(ctx, gradOutput):
        input, = ctx.saved_variables
        with torch.cuda.device_of(input):
            gradInput = Variable(input.data.new().resize_as_(input.data))
        if isinstance(input.data, torch.cuda.FloatTensor):
            with torch.cuda.device_of(input.data):
                encoding_lib.Encoding_Float_DilatedAvgPool2d_Backward(
                    gradInput.data, gradOutput.data,
                    ctx.kH, ctx.kW, ctx.dH, ctx.dW, ctx.padH, ctx.padW,
                    ctx.dilationH, ctx.dilationW)
        elif isinstance(input.data, torch.cuda.DoubleTensor):
            with torch.cuda.device_of(input.data):
                encoding_lib.Encoding_Double_DilatedAvgPool2d_Backward(
                    gradInput.data, gradOutput.data,
                    ctx.kH, ctx.kW, ctx.dH, ctx.dW, ctx.padH, ctx.padW,
                    ctx.dilationH, ctx.dilationW)
        else:
            raise RuntimeError('Unimplemented data type!')
        return gradInput, None, None, None, None


def dilatedavgpool2d(input, kernel_size, stride=None, padding=0,
                     dilation=1):
    """Dilated Average Pool 2d, for dilation of DenseNet.

    Reference:

        Hang Zhang, Kristin Dana, Jianping Shi, Zhongyue Zhang, Xiaogang Wang,
        Ambrish Tyagi, Amit Agrawal. “Context Encoding for Semantic Segmentation. CVPR 2018

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
    return _dilatedavgpool2d.apply(input, kernel_size, stride, padding, dilation)
