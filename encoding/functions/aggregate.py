##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## ECE Department, Rutgers University
## Email: zhang.hang@rutgers.edu
## Copyright (c) 2017
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import threading
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function, Variable
from .._ext import encoding_lib


class aggregate(Function):
    r"""
    Aggregate operation, aggregate the residuals of inputs (:math:`X`) with repect to the codewords (:math:`C`) with assignment weights (:math:`A`).
    

    .. math::
        e_{k} = \sum_{i=1}^{N} a_{ik} (x_i - d_k)

    Shape:
        - Input: :math:`A\in\mathcal{R}^{B\times N\times K}` :math:`X\in\mathcal{R}^{B\times N\times D}` :math:`C\in\mathcal{R}^{K\times D}`  (where :math:`B` is batch, :math:`N` is total number of features, :math:`K` is number is codewords, :math:`D` is feature dimensions.)
        - Output: :math:`E\in\mathcal{R}^{B\times K\times D}`

    Examples:
        >>> B,N,K,D = 2,3,4,5
        >>> A = Variable(torch.cuda.DoubleTensor(B,N,K).uniform_(-0.5,0.5), requires_grad=True)
        >>> X = Variable(torch.cuda.DoubleTensor(B,N,D).uniform_(-0.5,0.5), requires_grad=True)
        >>> C = Variable(torch.cuda.DoubleTensor(K,D).uniform_(-0.5,0.5), requires_grad=True)
        >>> func = encoding.aggregate()
        >>> E = func(A, X, C)

    """
    def forward(self, A, X, C):
        # A \in(BxNxK) R \in(BxNxKxD) => E \in(BxNxD)
        self.save_for_backward(A, X, C)
        B, N, K = A.size()
        D = X.size(2)
        with torch.cuda.device_of(A):
            E = A.new(B,K,D)
        if isinstance(A, torch.cuda.FloatTensor):
            with torch.cuda.device_of(A):
                encoding_lib.Encoding_Float_aggregateE_forward(E, A, X, C)
        elif isinstance(A, torch.cuda.DoubleTensor):
            with torch.cuda.device_of(A):
                encoding_lib.Encoding_Double_aggregateE_forward(E, A, X, C)
        else:
            raise RuntimeError('Unimplemented data type!')
        return E

    def backward(self, gradE):
        A, X, C = self.saved_tensors
        with torch.cuda.device_of(A):
            gradA = A.new().resize_as_(A)
            gradX = A.new().resize_as_(X)
            gradC = A.new().resize_as_(C)
        if isinstance(A, torch.cuda.FloatTensor):
            with torch.cuda.device_of(A):
                encoding_lib.Encoding_Float_aggregateE_backward(gradA, 
                    gradE, A, X, C)
        elif isinstance(A, torch.cuda.DoubleTensor):
            with torch.cuda.device_of(A):
                encoding_lib.Encoding_Double_aggregateE_backward(gradA, 
                    gradE, A, X, C)
        else:
            raise RuntimeError('Unimplemented data type!')
        gradX.copy_(torch.bmm(A, gradE))
        gradC.copy_((-gradE*A.sum(1).unsqueeze(2)).sum(0))
        return gradA, gradX, gradC


class scaledL2(Function):
    r"""
    scaledL2 distance

    .. math::
        sl_{ik} = s_k \|x_i-c_k\|^2

    Shape:
        - Input: :math:`X\in\mathcal{R}^{B\times N\times D}` :math:`C\in\mathcal{R}^{K\times D}` :math:`S\in \mathcal{R}^K` (where :math:`B` is batch, :math:`N` is total number of features, :math:`K` is number is codewords, :math:`D` is feature dimensions.)
        - Output: :math:`E\in\mathcal{R}^{B\times N\times K}`

    """
    def forward(self, X, C, S):
        B,N,D = X.size()
        K = C.size(0)
        with torch.cuda.device_of(X):
            SL = X.new(B,N,K)
        if isinstance(X, torch.cuda.FloatTensor):
            with torch.cuda.device_of(X):
                encoding_lib.Encoding_Float_scaledl2_forward(SL, X, C, S)
        elif isinstance(X, torch.cuda.DoubleTensor):
            with torch.cuda.device_of(X):
                encoding_lib.Encoding_Double_scaledl2_forward(SL, X, C, S)
        else:
            raise RuntimeError('Unimplemented data type!')
        self.save_for_backward(X, C, S, SL)
        return SL
    def backward(self, gradSL):
        X, C, S, SL = self.saved_tensors
        K = C.size(0)
        with torch.cuda.device_of(X):
            gradX = X.new().resize_as_(X)
            gradC = X.new().resize_as_(C)
            gradS = X.new().resize_as_(S)
        if isinstance(X, torch.cuda.FloatTensor):
            with torch.cuda.device_of(X):
                encoding_lib.Encoding_Float_scaledl2_backward(gradSL, 
                    gradX, gradC, X, C, S)
        elif isinstance(X, torch.cuda.DoubleTensor):
            with torch.cuda.device_of(X):
                encoding_lib.Encoding_Double_scaledl2_backward(gradSL, 
                    gradX, gradC, X, C, S)
        else:
            raise RuntimeError('Unimplemented data type!')
        gradS.copy_((gradSL*(SL/S.view(1,1,K))).sum(0).sum(0))
        return gradX, gradC, gradS


class aggregateP(Function):
    def forward(self, A, R):
        # A \in(BxNxK) R \in(BxNxKxD) => E \in(BxNxD)
        self.save_for_backward(A, R)
        B, N, K, D = R.size()
        with torch.cuda.device_of(A):
            E = A.new(B,K,D)
        if isinstance(A, torch.cuda.FloatTensor):
            with torch.cuda.device_of(A):
                encoding_lib.Encoding_Float_aggregate_forward(E, A, R)
        elif isinstance(A, torch.cuda.DoubleTensor):
            with torch.cuda.device_of(A):
                encoding_lib.Encoding_Double_aggregate_forward(E, A, R)
        else:
            raise RuntimeError('Unimplemented data type!')
        return E

    def backward(self, gradE):
        A, R = self.saved_tensors
        with torch.cuda.device_of(A):
            gradA = A.new().resize_as_(A)
            gradR = R.new().resize_as_(R)
        if isinstance(A, torch.cuda.FloatTensor):
            with torch.cuda.device_of(A):
                encoding_lib.Encoding_Float_aggregate_backward(gradA, 
                    gradR, gradE, A, R)
        elif isinstance(A, torch.cuda.DoubleTensor):
            with torch.cuda.device_of(A):
                encoding_lib.Encoding_Double_aggregate_backward(gradA, 
                    gradR, gradE, A, R)
        else:
            raise RuntimeError('Unimplemented data type!')
        return gradA, gradR


class residual(Function):
    r"""
    Calculate residuals over a mini-batch
    
    .. math::
        r_{ik} = x_i - c_k

    Shape:
        - Input: :math:`X\in\mathcal{R}^{B\times N\times D}` :math:`C\in\mathcal{R}^{K\times D}` (where :math:`B` is batch, :math:`N` is total number of features, :math:`K` is number is codewords, :math:`D` is feature dimensions.)
        - Output: :math:`R\in\mathcal{R}^{B\times N\times K\times D}`

    """
    def forward(self, X, C):
        # X \in(BxNxD) D \in(KxD) R \in(BxNxKxD) 
        B, N, D = X.size()
        K = C.size(0)
        with torch.cuda.device_of(X):
            R = X.new(B,N,K,D)
        if isinstance(X, torch.cuda.FloatTensor):
            with torch.cuda.device_of(X):
                encoding_lib.Encoding_Float_residual_forward(R, X, C)
        elif isinstance(X, torch.cuda.DoubleTensor):
            with torch.cuda.device_of(X):
                encoding_lib.Encoding_Double_residual_forward(R, X, C)
        else:
            raise RuntimeError('Unimplemented data type!')
        return R

    def backward(self, gradR):
        B, N, K, D = gradR.size()
        with torch.cuda.device_of(gradR):
            gradX = gradR.new(B,N,D)
            gradD = gradR.new(K,D)
        if isinstance(gradR, torch.cuda.FloatTensor):
            with torch.cuda.device_of(gradR):
                encoding_lib.Encoding_Float_residual_backward(gradR, 
                    gradX, gradD)
        elif isinstance(gradR, torch.cuda.DoubleTensor):
            with torch.cuda.device_of(gradR):
                encoding_lib.Encoding_Double_residual_backward(gradR, 
                    gradX, gradD)
        else:
            raise RuntimeError('Unimplemented data type!')
        return gradX, gradD


class square_squeeze(Function):
    def forward(self, R):
        B, N, K, D = R.size()
        with torch.cuda.device_of(R):
            L = R.new(B,N,K)
        if isinstance(R, torch.cuda.FloatTensor):
            with torch.cuda.device_of(R):
                encoding_lib.Encoding_Float_squaresqueeze_forward(L, R)
        elif isinstance(R, torch.cuda.DoubleTensor):
            with torch.cuda.device_of(R):
                encoding_lib.Encoding_Double_squaresqueeze_forward(L, R)
        else:
            raise RuntimeError('Unimplemented data type!')
        self.save_for_backward(L, R)
        return L

    def backward(self, gradL):
        L, R = self.saved_tensors
        B, N, K, D = R.size()
        with torch.cuda.device_of(R):
            gradR = R.new(B,N,K,D)
        if isinstance(R, torch.cuda.FloatTensor):
            with torch.cuda.device_of(gradL):
                encoding_lib.Encoding_Float_squaresqueeze_backward(gradL, 
                    gradR, R)
        elif isinstance(R, torch.cuda.DoubleTensor):
            with torch.cuda.device_of(gradL):
                encoding_lib.Encoding_Double_squaresqueeze_backward(gradL, 
                    gradR, R)
        else:
            raise RuntimeError('Unimplemented data type!')
        return gradR
    

def assign(R, S):
    L = square_squeeze()(R)
    K = S.size(0)
    SL = L * S.view(1,1,K)
    return F.softmax(SL)
