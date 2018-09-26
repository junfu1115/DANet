##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## Email: zhanghang0704@gmail.com
## Copyright (c) 2018
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

"""Functions for Encoding Layer"""
import torch
from torch.autograd import Function, Variable
import torch.nn.functional as F
from .. import lib

__all__ = ['aggregate', 'scaled_l2', 'pairwise_cosine']

class _aggregate(Function):
    @staticmethod
    def forward(ctx, A, X, C):
        # A \in(BxNxK) R \in(BxNxKxD) => E \in(BxNxD)
        ctx.save_for_backward(A, X, C)
        if A.is_cuda:
            E = lib.gpu.aggregate_forward(A, X, C)
        else:
            E = lib.cpu.aggregate_forward(A, X, C)
        return E

    @staticmethod
    def backward(ctx, gradE):
        A, X, C = ctx.saved_variables
        if A.is_cuda:
            gradA, gradX, gradC = lib.gpu.aggregate_backward(gradE, A, X, C)
        else:
            gradA, gradX, gradC = lib.cpu.aggregate_backward(gradE, A, X, C)
        return gradA, gradX, gradC

def aggregate(A, X, C):
    r""" Aggregate operation, aggregate the residuals of inputs (:math:`X`) with repect
    to the codewords (:math:`C`) with assignment weights (:math:`A`).

    .. math::

        e_{k} = \sum_{i=1}^{N} a_{ik} (x_i - d_k)

    Shape:
        - Input: :math:`A\in\mathcal{R}^{B\times N\times K}`
          :math:`X\in\mathcal{R}^{B\times N\times D}` :math:`C\in\mathcal{R}^{K\times D}`
          (where :math:`B` is batch, :math:`N` is total number of features,
          :math:`K` is number is codewords, :math:`D` is feature dimensions.)
        - Output: :math:`E\in\mathcal{R}^{B\times K\times D}`

    Examples:
        >>> B,N,K,D = 2,3,4,5
        >>> A = Variable(torch.cuda.DoubleTensor(B,N,K).uniform_(-0.5,0.5), requires_grad=True)
        >>> X = Variable(torch.cuda.DoubleTensor(B,N,D).uniform_(-0.5,0.5), requires_grad=True)
        >>> C = Variable(torch.cuda.DoubleTensor(K,D).uniform_(-0.5,0.5), requires_grad=True)
        >>> func = encoding.aggregate()
        >>> E = func(A, X, C)
    """
    return _aggregate.apply(A, X, C)

class _scaled_l2(Function):
    @staticmethod
    def forward(ctx, X, C, S):
        if X.is_cuda:
            SL = lib.gpu.scaled_l2_forward(X, C, S)
        else:
            SL = lib.cpu.scaled_l2_forward(X, C, S)
        ctx.save_for_backward(X, C, S, SL)
        return SL

    @staticmethod
    def backward(ctx, gradSL):
        X, C, S, SL = ctx.saved_variables
        if X.is_cuda:
            gradX, gradC, gradS = lib.gpu.scaled_l2_backward(gradSL, X, C, S, SL)
        else:
            gradX, gradC, gradS = lib.cpu.scaled_l2_backward(gradSL, X, C, S, SL)
        return gradX, gradC, gradS

def scaled_l2(X, C, S):
    r""" scaled_l2 distance

    .. math::
        sl_{ik} = s_k \|x_i-c_k\|^2

    Shape:
        - Input: :math:`X\in\mathcal{R}^{B\times N\times D}`
          :math:`C\in\mathcal{R}^{K\times D}` :math:`S\in \mathcal{R}^K`
          (where :math:`B` is batch, :math:`N` is total number of features,
          :math:`K` is number is codewords, :math:`D` is feature dimensions.)
        - Output: :math:`E\in\mathcal{R}^{B\times N\times K}`
    """
    return _scaled_l2.apply(X, C, S)

# Experimental
def pairwise_cosine(X, C, normalize=False):
    r"""Pairwise Cosine Similarity or Dot-product Similarity
    Shape:
        - Input: :math:`X\in\mathcal{R}^{B\times N\times D}`
          :math:`C\in\mathcal{R}^{K\times D}` :math:`S\in \mathcal{R}^K`
          (where :math:`B` is batch, :math:`N` is total number of features,
          :math:`K` is number is codewords, :math:`D` is feature dimensions.)
        - Output: :math:`E\in\mathcal{R}^{B\times N\times K}`
    """
    if normalize:
        X = F.normalize(X, dim=2, eps=1e-8)
        C = F.normalize(C, dim=1, eps=1e-8)
    return torch.matmul(X, C.t())
