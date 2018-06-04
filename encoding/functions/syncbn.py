##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## Email: zhanghang0704@gmail.com
## Copyright (c) 2018
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

"""Synchronized Cross-GPU Batch Normalization functions"""
import torch
from torch.autograd import Variable, Function
from .. import lib

__all__ = ['sum_square', 'batchnormtrain']

def sum_square(input):
    r"""Calculate sum of elements and sum of squares for Batch Normalization"""
    return _sum_square.apply(input)


class _sum_square(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        if input.is_cuda:
            xsum, xsqusum = lib.gpu.sumsquare_forward(input)
        else:
            raise NotImplemented
        return xsum, xsqusum

    @staticmethod
    def backward(ctx, gradSum, gradSquare):
        input, = ctx.saved_variables
        if input.is_cuda:
            gradInput = lib.gpu.sumsquare_backward(input, gradSum, gradSquare)
        else:
            raise NotImplemented
        return gradInput


class _batchnormtrain(Function):
    @staticmethod
    def forward(ctx, input, mean, std, gamma, beta):
        ctx.save_for_backward(input, mean, std, gamma, beta)
        if input.is_cuda:
            output = lib.gpu.batchnorm_forward(input, mean, std, gamma, beta)
        else:
            raise NotImplemented
        return output

    @staticmethod
    def backward(ctx, gradOutput):
        input, mean, std, gamma, beta = ctx.saved_variables
        if gradOutput.is_cuda:
            gradInput, gradMean, gradStd, gradGamma, gradBeta = \
                lib.gpu.batchnorm_backward(gradOutput, input, mean,
                                           std, gamma, beta, True)
        else:
            raise NotImplemented
        return gradInput, gradMean, gradStd, gradGamma, gradBeta


def batchnormtrain(input, mean, std, gamma, beta):
    r"""Applies Batch Normalization over a 3d input that is seen as a
    mini-batch.

    .. _encoding.batchnormtrain:

    .. math::

        y = \frac{x - \mu[x]}{ \sqrt{var[x] + \epsilon}} * \gamma + \beta

    Shape:
        - Input: :math:`(N, C)` or :math:`(N, C, L)`
        - Output: :math:`(N, C)` or :math:`(N, C, L)` (same shape as input)

    """
    return _batchnormtrain.apply(input, mean, std, gamma, beta)
