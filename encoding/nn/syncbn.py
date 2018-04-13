##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## ECE Department, Rutgers University
## Email: zhang.hang@rutgers.edu
## Copyright (c) 2017
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

"""Synchronized Cross-GPU Batch Normalization Module"""
import threading
import torch
from torch.nn import Module, Sequential, Conv1d, Conv2d, ConvTranspose2d, \
    ReLU, Sigmoid, MaxPool2d, AvgPool2d, AdaptiveAvgPool2d, Dropout2d, Linear
from torch.nn.parameter import Parameter

from ..functions import batchnormtrain, batchnormeval, sum_square
from ..parallel import allreduce

# import standard layers for convinent use
__all__ = ['BatchNorm1d', 'BatchNorm2d', 'Module', 'Sequential', 'Conv1d',
           'Conv2d', 'ConvTranspose2d', 'ReLU', 'Sigmoid', 'MaxPool2d',
           'AvgPool2d', 'AdaptiveAvgPool2d', 'Dropout2d', 'Linear']

class BatchNorm1d(Module):
    r"""Cross-GPU Synchronized Batch normalization (SyncBN)

    Standard BN [1]_ implementation only normalize the data within each device.
    SyncBN normalizes the input within the whole mini-batch.
    We follow the sync-onece implmentation described in the paper [2]_ .

    .. math::

        y = \frac{x - mean[x]}{ \sqrt{Var[x] + \epsilon}} * gamma + beta

    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and gamma and beta are learnable parameter vectors
    of size C (where C is the input size).

    During training, this layer keeps a running estimate of its computed mean
    and variance. The running sum is kept with a default momentum of 0.1.

    During evaluation, this running mean/variance is used for normalization.

    Because the BatchNorm is done over the `C` dimension, computing statistics
    on `(N, L)` slices, it's common terminology to call this Temporal BatchNorm

    Args:
        num_features: num_features from an expected input of size
            `batch_size x num_features [x width]`
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Default: 0.1
        affine: a boolean value that when set to ``True``, gives the layer learnable
            affine parameters. Default: ``True``

    Shape:
        - Input: :math:`(N, C)` or :math:`(N, C, L)`
        - Output: :math:`(N, C)` or :math:`(N, C, L)` (same shape as input)

    Examples:
        >>> # Use exactly the same as standard BatchNrom1d
        >>> m = nn.BatchNorm1d(100)
        >>> output = m(input)
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super(BatchNorm1d, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps
        self.momentum = momentum
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features))
            self.bias = Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.reset_parameters()
        self.writelock = threading.Lock()
        nGPUs = torch.cuda.device_count()
        self.xsum = SharedTensor(nGPUs)
        self.xsquare = SharedTensor(nGPUs)

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)
        if self.affine:
            self.weight.data.uniform_()
            self.bias.data.zero_()

    def __repr__(self):
        return ('{name}({num_features}, eps={eps}, momentum={momentum},'
                ' affine={affine})'
                .format(name=self.__class__.__name__, **self.__dict__))

    def _check_input_dim(self, input):
        if input.dim() != 3:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, input):
        self._check_input_dim(input)
        if self.training:
            # push the value
            isum, isquare = sum_square(input.unsqueeze(3))
            idxs = self.xsum.push(isum)
            idxq = self.xsquare.push(isquare)
            xsum = self.xsum[idxs]
            xsquare = self.xsquare[idxq]
            # calculate N
            N = len(self.xsum)*input.size(0)*input.size(2)
            mean = xsum / N
            sumvar = xsquare - xsum * xsum / N
            unbias_var = sumvar / (N - 1)
            std = (sumvar / N + self.eps).sqrt()
            # update running_mean and var
            self.running_mean = (1-self.momentum) * self.running_mean \
                + self.momentum * mean.data
            self.running_var = (1-self.momentum) * self.running_var + \
                self.momentum * unbias_var.data
            # forward
            return batchnormtrain(input, self.weight,
                                  self.bias, mean, std)
        else:
            std = (self.running_var + self.eps).sqrt()
            return batchnormeval(input, self.weight, self.bias,
                                 self.running_mean, std)


class BatchNorm2d(Module):
    r"""Cross-GPU Synchronized Batch normalization (SyncBN)

    Standard BN [1]_ implementation only normalize the data within each device.
    SyncBN normalizes the input within the whole mini-batch.
    We follow the sync-onece implmentation described in the paper [2]_ .

    .. math::

        y = \frac{x - mean[x]}{ \sqrt{Var[x] + \epsilon}} * gamma + beta

    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and gamma and beta are learnable parameter vectors
    of size C (where C is the input size).

    During training, this layer keeps a running estimate of its computed mean
    and variance. The running sum is kept with a default momentum of 0.1.

    During evaluation, this running mean/variance is used for normalization.

    Because the BatchNorm is done over the `C` dimension, computing statistics
    on `(N, H, W)` slices, it's common terminology to call this Spatial BatchNorm

    Args:
        num_features: num_features from an expected input of
            size batch_size x num_features x height x width
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Default: 0.1
        affine: a boolean value that when set to ``True``, gives the layer learnable
            affine parameters. Default: ``True``

    Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)` (same shape as input)

    Reference:
        .. [1] Ioffe, Sergey, and Christian Szegedy. "Batch normalization: Accelerating deep network training by reducing internal covariate shift." *ICML 2015*
        .. [2] Hang Zhang, Kristin Dana, Jianping Shi, Zhongyue Zhang, Xiaogang Wang, Ambrish Tyagi, and Amit Agrawal. "Context Encoding for Semantic Segmentation." *CVPR 2018*

    Examples:
        >>> # Use exactly the same as standard BatchNrom2d
        >>> m = nn.BatchNorm2d(100)
        >>> output = m(input)
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super(BatchNorm2d, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps
        self.momentum = momentum
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features))
            self.bias = Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.reset_parameters()
        self.writelock = threading.Lock()
        nGPUs = torch.cuda.device_count()
        self.xsum, self.xsquare = SharedTensor(nGPUs), SharedTensor(nGPUs)

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)
        if self.affine:
            self.weight.data.uniform_()
            self.bias.data.zero_()

    def __repr__(self):
        return ('{name}({num_features}, eps={eps}, momentum={momentum},'
                ' affine={affine})'
                .format(name=self.__class__.__name__, **self.__dict__))

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, input):
        self._check_input_dim(input)
        if self.training:
            # push the value
            isum, isquare = sum_square(input)
            idxs = self.xsum.push(isum)
            idxq = self.xsquare.push(isquare)
            xsum = self.xsum[idxs]
            xsquare = self.xsquare[idxq]
            # calculate N
            N = len(self.xsum)*input.size(0)*input.size(2)*input.size(3)
            mean = xsum / N
            sumvar = xsquare - xsum * xsum / N
            unbias_var = sumvar / (N - 1)
            std = (sumvar / N + self.eps).sqrt()
            # update running_mean and var
            self.running_mean = (1-self.momentum) * self.running_mean \
                + self.momentum * mean.data
            self.running_var = (1-self.momentum) * self.running_var + \
                self.momentum * unbias_var.data
            # forward
            B, C, H, W = input.size()
            output = batchnormtrain(
                input.view(B, C, -1).contiguous(), self.weight,
                self.bias, mean,
                std)
            return output.view(B, C, H, W)
        else:
            std = (self.running_var + self.eps).sqrt()
            B, C, H, W = input.size()
            return batchnormeval(input.view(B, C, -1).contiguous(), self.weight, self.bias,
                                 self.running_mean, std).view(B, C, H, W)


class SharedTensor(object):
    """Shared Tensor
    """
    def __init__(self, nGPUs):
        self.mutex = threading.Lock()
        self.all_tasks_done = threading.Condition(self.mutex)
        self.nGPUs = nGPUs
        self._clear()

    def _clear(self):
        self.list = []
        self.push_tasks = self.nGPUs
        self.reduce_tasks = self.nGPUs

    def push(self, t):
        """push a Tensor
        """
        with self.mutex:
            if self.push_tasks == 0:
                self._clear()
            self.list.append(t)
            idx = len(self.list) - 1
            self.push_tasks -= 1

        with self.all_tasks_done:
            if self.push_tasks == 0:
                self.all_tasks_done.notify_all()
            while self.push_tasks:
                self.all_tasks_done.wait()
        return idx

    def _reduce(self):
        with self.mutex:
            if self.reduce_tasks == self.nGPUs:
                assert(len(self.list) == self.nGPUs)
                self.outlist = allreduce(*self.list)
                self.reduce_tasks -= 1
            else:
                self.reduce_tasks -= 1

        with self.all_tasks_done:
            if self.reduce_tasks == 0:
                self.all_tasks_done.notify_all()
            while self.reduce_tasks:
                self.all_tasks_done.wait()

    def __getitem__(self, idx):
        self._reduce()
        return self.outlist[idx]

    def __len__(self):
        return len(self.list)

    def __repr__(self):
        return ('SharedTensor')
