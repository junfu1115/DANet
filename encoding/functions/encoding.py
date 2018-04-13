##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## ECE Department, Rutgers University
## Email: zhang.hang@rutgers.edu
## Copyright (c) 2017
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

"""Functions for Encoding Layer"""
import torch
from torch.autograd import Function, Variable
from .._ext import encoding_lib

__all__ = ['aggregate', 'scaledL2']

class _aggregate(Function):
    @staticmethod
    def forward(ctx, A, X, C):
        # A \in(BxNxK) R \in(BxNxKxD) => E \in(BxNxD)
        ctx.save_for_backward(A, X, C)
        B, _, K = A.size()
        D = X.size(2)
        with torch.cuda.device_of(A):
            E = A.new(B, K, D)
        if isinstance(A, torch.cuda.FloatTensor):
            with torch.cuda.device_of(A):
                encoding_lib.Encoding_Float_aggregate_forward(E, A, X, C)
        elif isinstance(A, torch.cuda.DoubleTensor):
            with torch.cuda.device_of(A):
                encoding_lib.Encoding_Double_aggregate_forward(E, A, X, C)
        else:
            raise RuntimeError('Unimplemented data type!')
        return E

    @staticmethod
    def backward(ctx, gradE):
        A, X, C = ctx.saved_variables
        with torch.cuda.device_of(A):
            gradA = Variable(A.data.new().resize_as_(A.data))
            gradX = Variable(A.data.new().resize_as_(X.data))
            gradC = Variable(A.data.new().resize_as_(C.data))
        if isinstance(A.data, torch.cuda.FloatTensor):
            with torch.cuda.device_of(A.data):
                encoding_lib.Encoding_Float_aggregate_backward(gradA.data, \
                    gradE.data, A.data, X.data, C.data)
        elif isinstance(A.data, torch.cuda.DoubleTensor):
            with torch.cuda.device_of(A.data):
                encoding_lib.Encoding_Double_aggregate_backward(gradA.data, \
                    gradE.data, A.data, X.data, C.data)
        else:
            raise RuntimeError('Unimplemented data type!')
        gradX.data.copy_(torch.bmm(A, gradE).data)
        gradC.data.copy_((-gradE*A.sum(1).unsqueeze(2)).sum(0).data)
        return gradA, gradX, gradC

def aggregate(A, X, C):
    r"""
    Aggregate operation, aggregate the residuals of inputs (:math:`X`) with repect
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

class _scaledL2(Function):
    @staticmethod
    def forward(ctx, X, C, S):
        B, N, _ = X.size()
        K = C.size(0)
        with torch.cuda.device_of(X):
            SL = X.new(B, N, K)
        if isinstance(X, torch.cuda.FloatTensor):
            with torch.cuda.device_of(X):
                encoding_lib.Encoding_Float_scaledl2_forward(SL, X, C, S)
        elif isinstance(X, torch.cuda.DoubleTensor):
            with torch.cuda.device_of(X):
                encoding_lib.Encoding_Double_scaledl2_forward(SL, X, C, S)
        else:
            raise RuntimeError('Unimplemented data type!')
        ctx.save_for_backward(X, C, S, SL)
        return SL

    @staticmethod
    def backward(ctx, gradSL):
        X, C, S, SL = ctx.saved_variables
        K = C.size(0)
        with torch.cuda.device_of(X.data):
            gradX = Variable(X.data.new().resize_as_(X.data))
            gradC = Variable(X.data.new().resize_as_(C.data))
            gradS = Variable(X.data.new().resize_as_(S.data))
        if isinstance(X.data, torch.cuda.FloatTensor):
            with torch.cuda.device_of(X.data):
                encoding_lib.Encoding_Float_scaledl2_backward(gradSL.data, \
                    gradX.data, gradC.data, X.data, C.data, S.data)
        elif isinstance(X.data, torch.cuda.DoubleTensor):
            with torch.cuda.device_of(X.data):
                encoding_lib.Encoding_Double_scaledl2_backward(gradSL.data, \
                    gradX.data, gradC.data, X.data, C.data, S.data)
        else:
            raise RuntimeError('Unimplemented data type!')
        gradS.data.copy_((gradSL*(SL/S.view(1, 1, K))).sum(0).sum(0).data)
        return gradX, gradC, gradS


def scaledL2(X, C, S):
    r"""
    scaledL2 distance

    .. math::
        sl_{ik} = s_k \|x_i-c_k\|^2

    Shape:
        - Input: :math:`X\in\mathcal{R}^{B\times N\times D}`
          :math:`C\in\mathcal{R}^{K\times D}` :math:`S\in \mathcal{R}^K`
          (where :math:`B` is batch, :math:`N` is total number of features,
          :math:`K` is number is codewords, :math:`D` is feature dimensions.)
        - Output: :math:`E\in\mathcal{R}^{B\times N\times K}`

    """
    return _scaledL2.apply(X, C, S)
