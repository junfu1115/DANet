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
from ..functions import *


class Encoding(nn.Module):
    r"""
    Encoding Layer: learnable residual encoders over 3d or 4d input that is seen as a mini-batch.

    .. math::

        a_{ik} = \frac{exp(-\beta\|x_{i}-c_k\|^2)}{\sum_{j=1}^K exp(-\beta\|x_{i}-c_j\|^2)}

    Args:
        D: dimention of the features or feature channels
        K: number of codeswords

    Shape:
        - Input: :math:`X\in\mathcal{R}^{B\times N\times D}` or :math:`\mathcal{R}^{B\times D\times H\times W}` (where :math:`B` is batch, :math:`N` is total number of features or :math:`H\times W`.)
        - Output: :math:`E\in\mathcal{R}^{B\times K\times D}`
        
    Attributes:
        codewords (Tensor): the learnable codewords of shape (:math:`K\times D`)
        scale (Tensor): the learnable scale factor of visual centers

    Examples:
        >>> import encoding
        >>> import torch
        >>> import torch.nn.functional as F
        >>> from torch.autograd import Variable, gradcheck
        >>> B,C,H,W,K = 2,3,4,5,6
        >>> X = Variable(torch.cuda.DoubleTensor(B,C,H,W).uniform_(-0.5,0.5), requires_grad=True)
        >>> layer = encoding.Encoding(C,K).double().cuda()
        >>> E = layer(X)

    Reference:
        Hang Zhang, Jia Xue, and Kristin Dana. "Deep TEN: Texture Encoding Network." *The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) 2017*
    """
    def __init__(self, D, K):
        super(Encoding, self).__init__()
        # init codewords and smoothing factor
        self.D, self.K = D, K
        self.codewords = nn.Parameter(torch.Tensor(K, D), 
            requires_grad=True)
        self.scale = nn.Parameter(torch.Tensor(K), requires_grad=True) 
        self.reset_params()
        
    def reset_params(self):
        std1 = 1./((self.K*self.D)**(1/2))
        std2 = 1./((self.K)**(1/2))
        self.codewords.data.uniform_(-std1, std1)
        self.scale.data.uniform_(-std2, std2)

    def forward(self, X):
        # input X is a 4D tensor
        assert(X.size(1)==self.D,"Encoding Layer wrong channels!")
        if X.dim() == 3:
            # BxDxN
            B, N, K, D = X.size(0), X.size(2), self.K, self.D
            X = X.transpose(1,2).contiguous()
        elif X.dim() == 4:
            # BxDxHxW
            B, N, K, D = X.size(0), X.size(2)*X.size(3), self.K, self.D
            X = X.view(B,D,-1).transpose(1,2).contiguous()
        else:
            raise RuntimeError('Encoding Layer unknown input dims!')
        # assignment weights
        A = F.softmax(scaledL2()(X, self.codewords, self.scale))
        # aggregate
        E = aggregate()(A, X, self.codewords)
        return E

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'N x ' + str(self.D) + '=>' + str(self.K) + 'x' \
            + str(self.D) + ')'


class Aggregate(nn.Module):
    r"""
    Aggregate operation, aggregate the residuals (:math:`R`) with assignment weights (:math:`A`).

    .. math::
        e_{k} = \sum_{i=1}^{N} a_{ik} (r_{ik})

    Shape:
        - Input: :math:`A\in\mathcal{R}^{B\times N\times K}` :math:`R\in\mathcal{R}^{B\times N\times K\times D}` (where :math:`B` is batch, :math:`N` is total number of features, :math:`K` is number is codewords, :math:`D` is feature dimensions.)
        - Output: :math:`E\in\mathcal{R}^{B\times K\times D}`

    """ 
    def forward(self, A, R):
        return aggregateP()(A, R)


class EncodingP(nn.Module):
    def __init__(self, D, K):
        super(EncodingP, self).__init__()
        # init codewords and smoothing factor
        self.D, self.K = D, K
        self.codewords = nn.Parameter(torch.Tensor(K, D), 
            requires_grad=True)
        self.scale = nn.Parameter(torch.Tensor(K), requires_grad=True) 
        self.reset_params()
        print('EncodingP is deprecated, please use Encoding.')
        
    def reset_params(self):
        std1 = 1./((self.K*self.D)**(1/2))
        std2 = 1./((self.K)**(1/2))
        self.codewords.data.uniform_(-std1, std1)
        self.scale.data.uniform_(-std2, std2)

    def forward(self, X):
        # input X is a 4D tensor
        assert(X.size(1)==self.D,"Encoding Layer wrong channels!")
        if X.dim() == 3:
            # BxDxN
            B, N, K, D = X.size(0), X.size(2), self.K, self.D
            X = X.transpose(1,2)
        elif X.dim() == 4:
            # BxDxHxW
            B, N, K, D = X.size(0), X.size(2)*X.size(3), self.K, self.D
            X = X.view(B,D,-1).transpose(1,2)
        else:
            raise RuntimeError('Encoding Layer unknown input dims!')
        # calculate residuals
        R = residual()(X.contiguous(), self.codewords)
        # assignment weights
        A = assign(R, self.scale)
        # aggregate
        E = aggregateP()(A, R)

        return E

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'N x ' + str(self.D) + '=>' + str(self.K) + 'x' \
            + str(self.D) + ')'


