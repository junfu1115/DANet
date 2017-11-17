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
from ..functions import scaledL2, aggregate
from ..parallel import my_data_parallel

__all__ = ['Encoding', 'EncodingShake', 'Inspiration', 'GramMatrix'] 

class Encoding(nn.Module):
    r"""
    Encoding Layer: a learnable residual encoder over 3d or 4d input that 
    is seen as a mini-batch.

    .. image:: _static/img/cvpr17.svg
        :width: 50%
        :align: center

    .. math::

        e_{ik} = \frac{exp(-s_k\|x_{i}-c_k\|^2)}{\sum_{j=1}^K exp(-s_j\|x_{i}-c_j\|^2)} (x_i - c_k)

    Please see the `example of training Deep TEN <./experiments/texture.html>`_.

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
        self.codewords.data.uniform_(-std1, std1)
        self.scale.data.uniform_(-1, 0)

    def forward(self, X):
        if isinstance(X, tuple) or isinstance(X, list):
            # for self-parallel mode, please see encoding.nn
            return my_data_parallel(self, X)
        elif not isinstance(X, Variable):
            raise RuntimeError('unknown input type')
        # input X is a 4D tensor
        assert(X.size(1)==self.D)
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
        #A = F.softmax(scaledL2(X, self.codewords, self.scale).view(B*N,-1), dim=1).view(B,N,K)
        A = F.softmax(scaledL2(X, self.codewords, self.scale), dim=2)
        # aggregate
        E = aggregate(A, X, self.codewords)
        return E

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'N x ' + str(self.D) + '=>' + str(self.K) + 'x' \
            + str(self.D) + ')'

class EncodingShake(nn.Module):
    def __init__(self, D, K):
        super(EncodingShake, self).__init__()
        # init codewords and smoothing factor
        self.D, self.K = D, K
        self.codewords = nn.Parameter(torch.Tensor(K, D), 
            requires_grad=True)
        self.scale = nn.Parameter(torch.Tensor(K), requires_grad=True) 
        self.reset_params()
        
    def reset_params(self):
        std1 = 1./((self.K*self.D)**(1/2))
        self.codewords.data.uniform_(-std1, std1)
        self.scale.data.uniform_(-1, 0)

    def shake(self):
        if self.training:
            self.scale.data.uniform_(-1, 0)
        else:
            self.scale.data.zero_().add_(-0.5)

    def forward(self, X):
        if isinstance(X, tuple) or isinstance(X, list):
            # for self-parallel mode, please see encoding.nn
            return my_data_parallel(self, X)
        elif not isinstance(X, Variable):
            raise RuntimeError('unknown input type')
        # input X is a 4D tensor
        assert(X.size(1)==self.D)
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
        # shake
        self.shake()
        # assignment weights
        A = F.softmax(scaledL2(X, self.codewords, self.scale).view(B*N,-1), dim=1).view(B,N,K)
        # aggregate
        E = aggregate(A, X, self.codewords)
        # shake
        self.shake()
        return E

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'N x ' + str(self.D) + '=>' + str(self.K) + 'x' \
            + str(self.D) + ')'


class Inspiration(nn.Module):
    r""" 
    Inspiration Layer (CoMatch Layer) enables the multi-style transfer in feed-forward network, which learns to match the target feature statistics during the training. 
    This module is differentialble and can be inserted in standard feed-forward network to be learned directly from the loss function without additional supervision. 

    .. math::
        Y = \phi^{-1}[\phi(\mathcal{F}^T)W\mathcal{G}]

    Please see the `example of MSG-Net <./experiments/style.html>`_  
    training multi-style generative network for real-time transfer.

    Reference:
        Hang Zhang and Kristin Dana. "Multi-style Generative Network for Real-time Transfer."  *arXiv preprint arXiv:1703.06953 (2017)*
    """
    def __init__(self, C, B=1):
        super(Inspiration, self).__init__()
        # B is equal to 1 or input mini_batch
        self.weight = nn.Parameter(torch.Tensor(1,C,C), requires_grad=True)
        # non-parameter buffer
        self.G = Variable(torch.Tensor(B,C,C), requires_grad=True)
        self.C = C
        self.reset_parameters()

    def reset_parameters(self):
        self.weight.data.uniform_(0.0, 0.02)

    def setTarget(self, target):
        self.G = target

    def forward(self, X):
        # input X is a 3D feature map
        self.P = torch.bmm(self.weight.expand_as(self.G),self.G)
        return torch.bmm(self.P.transpose(1,2).expand(X.size(0), self.C, self.C), X.view(X.size(0),X.size(1),-1)).view_as(X)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'N x ' + str(self.C) + ')'


class GramMatrix(nn.Module):
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

