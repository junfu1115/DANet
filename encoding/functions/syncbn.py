##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## ECE Department, Rutgers University
## Email: zhang.hang@rutgers.edu
## Copyright (c) 2017
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

"""Synchronized Cross-GPU Batch Normalization functions"""
import torch
from torch.autograd import Variable, Function
from .._ext import encoding_lib

__all__ = ['sum_square', 'batchnormtrain', 'batchnormeval']

def sum_square(input):
    r"""Calculate sum of elements and sum of squares for Batch Normalization"""
    return _sum_square.apply(input)


class _sum_square(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        C = input.size(1)
        with torch.cuda.device_of(input):
            xsum = input.new().resize_(C).zero_()
            xsquare = input.new().resize_(C).zero_()
        if isinstance(input, torch.cuda.FloatTensor):
            with torch.cuda.device_of(input):
                encoding_lib.Encoding_Float_sum_square_Forward(
                    input, xsum, xsquare)
        elif isinstance(input, torch.cuda.DoubleTensor):
            with torch.cuda.device_of(input):
                encoding_lib.Encoding_Double_sum_square_Forward(
                    input, xsum, xsquare)
        else:
            raise RuntimeError('Unimplemented data type!', type(input))
        return xsum, xsquare

    @staticmethod
    def backward(ctx, gradSum, gradSquare):
        input, = ctx.saved_variables
        with torch.cuda.device_of(input.data):
            gradInput = Variable(input.data.new().resize_as_(input.data).zero_())
        if isinstance(input.data, torch.cuda.FloatTensor):
            with torch.cuda.device_of(input.data):
                encoding_lib.Encoding_Float_sum_square_Backward(
                    gradInput.data, input.data, gradSum.data, gradSquare.data)
        elif isinstance(input.data, torch.cuda.DoubleTensor):
            with torch.cuda.device_of(input.data):
                encoding_lib.Encoding_Double_sum_square_Backward(
                    gradInput.data, input.data, gradSum.data, gradSquare.data)
        else:
            raise RuntimeError('Unimplemented data type!')
        return gradInput


class _batchnorm(Function):
    def __init__(self, training=False):
        super(_batchnorm, self).__init__()
        self.training = training

    def forward(self, input, gamma, beta, mean, std):
        self.save_for_backward(input, gamma, beta, mean, std)
        assert(input.dim() == 3)
        with torch.cuda.device_of(input):
            invstd = 1.0 / std
            output = input.new().resize_as_(input)
        if isinstance(input, torch.cuda.FloatTensor):
            with torch.cuda.device_of(input):
                encoding_lib.Encoding_Float_batchnorm_Forward(output, \
                    input, mean, invstd, gamma, beta)
        elif isinstance(input, torch.cuda.DoubleTensor):
            with torch.cuda.device_of(input):
                encoding_lib.Encoding_Double_batchnorm_Forward(output, \
                    input, mean, invstd, gamma, beta)
        else:
            raise RuntimeError('Unimplemented data type!')
        return output

    def backward(self, gradOutput):
        input, gamma, beta, mean, std = self.saved_tensors
        invstd = 1.0 / std
        with torch.cuda.device_of(input):
            gradInput = gradOutput.new().resize_as_(input).zero_()
            gradGamma = gradOutput.new().resize_as_(gamma).zero_()
            gradBeta = gradOutput.new().resize_as_(beta).zero_()
            gradMean = gradOutput.new().resize_as_(mean).zero_()
            gradStd = gradOutput.new().resize_as_(std).zero_()

        if isinstance(input, torch.cuda.FloatTensor):
            with torch.cuda.device_of(input):
                encoding_lib.Encoding_Float_batchnorm_Backward(
                    gradOutput, input, gradInput, gradGamma, gradBeta,
                    mean, invstd, gamma, beta, gradMean, gradStd,
                    self.training)
        elif isinstance(input, torch.cuda.DoubleTensor):
            with torch.cuda.device_of(input):
                encoding_lib.Encoding_Double_batchnorm_Backward(
                    gradOutput, input, gradInput, gradGamma, gradBeta,
                    mean, invstd, gamma, beta, gradMean, gradStd,
                    self.training)
        else:
            raise RuntimeError('Unimplemented data type!')
        return gradInput, gradGamma, gradBeta, gradMean, gradStd


def batchnormtrain(input, gamma, beta, mean, std):
    r"""Applies Batch Normalization over a 3d input that is seen as a
    mini-batch.

    .. _encoding.batchnormtrain:

    .. math::

        y = \frac{x - \mu[x]}{ \sqrt{var[x] + \epsilon}} * \gamma + \beta

    Shape:
        - Input: :math:`(N, C)` or :math:`(N, C, L)`
        - Output: :math:`(N, C)` or :math:`(N, C, L)` (same shape as input)

    """
    return _batchnorm(True)(input, gamma, beta, mean, std)


def batchnormeval(input, gamma, beta, mean, std):
    r"""Applies Batch Normalization over a 3d input that is seen as a
    mini-batch.

    Please see encoding.batchnormtrain_
    """
    return _batchnorm(False)(input, gamma, beta, mean, std)

