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
import torch.cuda.comm as comm
from torch.autograd import Variable
from torch.nn import Module, Sequential
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.utils import _single, _pair, _triple
from torch.nn.parallel.scatter_gather import scatter, scatter_kwargs, \
    gather

from ..functions import view_each, multi_each, sum_each, batchnormtrain, batchnormeval, sum_square 
from ..parallel import my_data_parallel, Broadcast, AllReduce

__all__ = ['BatchNorm1d', 'BatchNorm2d']

class BatchNorm1d(Module):
    r"""Synchronized Batch Normalization 1d

    `Implementation ideas <./notes/syncbn.html>`_. Please use compatible :class:`encoding.parallel.SelfDataParallel` and :class:`encoding.nn`

    Reference::
        We provide this code for a comming paper.

    Applies Batch Normalization over a 2d or 3d input that is seen as a
    mini-batch.

    .. math::

        y = \frac{x - \mu[x]}{ \sqrt{var[x] + \epsilon}} * \gamma + \beta

    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and gamma and beta are learnable parameter vectors
    of size C (where C is the input size).

    During training, this layer keeps a running estimate of its computed mean
    and variance. The running sum is kept with a default momentum of 0.1.

    During evaluation, this running mean/variance is used for normalization.

    Args:
        num_features: num_features from an expected input of size
            `batch_size x num_features [x width]`
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Default: 0.1
        affine: a boolean value that when set to true, gives the layer 
            learnable affine parameters. Default: True

    Shape:
        - Input: :math:`(N, C)` or :math:`(N, C, L)`
        - Output: :math:`(N, C)` or :math:`(N, C, L)` (same shape as input)

    Examples:
        >>> m = encoding.nn.BatchNorm1d(100).cuda()
        >>> input = autograd.Variable(torch.randn(20, 100)).cuda()
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
            raise ValueError('expected 3D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, input):
        if isinstance(input, Variable):
            self._check_input_dim(input)
            if self.training:
                xsum, xsquare = sum_square(input.unsqueeze(3))
                N = input.size(0)*input.size(2)
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
                output = batchnormtrain(
                    input, self.weight, 
                    self.bias, mean, 
                    std)
                return output
            else:
                var_mean = Variable(self.running_mean, requires_grad=False)
                bias_var = Variable(self.running_var, requires_grad=False)
                std = (bias_var + self.eps).sqrt()
                return batchnormeval(
                    input, self.weight, self.bias, var_mean, std)

        elif isinstance(input, tuple) or isinstance(input, list):
            self._check_input_dim(input[0])
            # if evaluation, do it simple
            if not self.training:
                return my_data_parallel(self, input)
            if len(input) == 1:
                return self.forward(input[0])
            # calculate mean and var using multithreading
            all_sum, all_xsquare = {},{}
            def _worker(i, x, lock):
                try:
                    with torch.cuda.device_of(x):
                        xsum, xsquare = sum_square(x.unsqueeze(3))
                    with lock:
                        all_sum[i] = xsum 
                        all_xsquare[i] = xsquare 
                except Exception as e:
                    with lock:
                        all_sum[i] = e
                        all_xsquare[i] = e
            threads = [threading.Thread(target=_worker,
                                        args=(i, x, self.writelock))
                        for i, x in enumerate(input)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            # convert to list
            def _to_list(x):
                outputs = []
                for i in range(len(x)):
                    outputs.append(x[i])
                return outputs
            
            all_sum = _to_list(all_sum)
            all_xsquare = _to_list(all_xsquare)
            xsums = AllReduce()(*all_sum)
            xsquares = AllReduce()(*all_xsquare)

            nGPUs = len(input)
            N = nGPUs * input[0].size(0)*input[0].size(2)
            assert(N>1)
            xmean = xsums[0].data / N
            unbias_var = (xsquares[0].data - N * xmean * xmean) / (N-1) 
            # update running_mean and var
            self.running_mean = (1-self.momentum) * self.running_mean \
                + self.momentum * xmean
            self.running_var = (1-self.momentum) * self.running_var + \
                self.momentum * unbias_var
            # Broadcast the weight, bias, mean, std
            device_ids = list(range(torch.cuda.device_count()))
            weights = Broadcast(device_ids[:len(input)])(self.weight) 
            biases = Broadcast(device_ids[:len(input)])(self.bias)
            # parallel-apply
            results = {}
            def _worker_bn(i, x, xsum, xsquare, weight, bias, lock):
                var_input = _get_a_var(x)
                mean = xsum / N
                std  = (xsquare / N - mean * mean + self.eps).sqrt()
                try:
                    with torch.cuda.device_of(var_input):
                        result = batchnormtrain(
                            x, weight, bias, mean, std)
                    with lock: 
                        results[i] = result
                except Exception as e:
                    with lock:
                        results[i] = e
            threads = [threading.Thread(target=_worker_bn,
                                        args=(i, x, xsum, xsquare, weight, 
                                              bias, self.writelock)
                                       )
                        for i,( x, xsum, xsquare, weight, bias) in 
                        enumerate(zip(input, xsums, xsquares, 
                                      weights, biases))]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            outputs = []
            for i in range(len(results)):
                output = results[i]
                if isinstance(output, Exception):
                    raise output
                outputs.append(output)
            return outputs
        else:
            raise RuntimeError('unknown input type')


class BatchNorm2d(Module):
    r"""Synchronized Batch Normalization 2d

    `Implementation ideas <./notes/syncbn.html>`_. Please use compatible :class:`encoding.parallel.SelfDataParallel` and :class:`encoding.nn`. 

    Reference::
        We provide this code for a comming paper.

    Applies Batch Normalization over a 4d input that is seen as a mini-batch
    of 3d inputs

    .. math::

        y = \frac{x - \mu[x]}{ \sqrt{var[x] + \epsilon}} * \gamma + \beta

    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and gamma and beta are learnable parameter vectors
    of size C (where C is the input size).

    During training, this layer keeps a running estimate of its computed mean
    and variance. The running sum is kept with a default momentum of 0.1.

    During evaluation, this running mean/variance is used for normalization.

    Args:
        num_features: num_features from an expected input of
            size batch_size x num_features x height x width
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Default: 0.1
        affine: a boolean value that when set to true, gives the layer learnable
            affine parameters. Default: True

    Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)` (same shape as input)

    Examples:
        >>> m = encoding.nn.BatchNorm2d(100).cuda()
        >>> input = autograd.Variable(torch.randn(20, 100, 35, 45)).cuda()
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
        if isinstance(input, Variable):
            self._check_input_dim(input)
            if self.training:
                xsum, xsquare = sum_square(input)
                N = input.size(0)*input.size(2)*input.size(3)
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
                    input.view(B,C,-1).contiguous(), self.weight, 
                    self.bias, mean, 
                    std)
                return output.view(B, C, H, W)
            else:
                var_mean = Variable(self.running_mean, requires_grad=False)
                bias_var = Variable(self.running_var, requires_grad=False)
                std = (bias_var + self.eps).sqrt()
                B, C, H, W = input.size()
                return batchnormeval(
                    input.view(B,C,-1).contiguous(), 
                    self.weight, self.bias, var_mean, 
                    std).view(B, C, H, W)

        elif isinstance(input, tuple) or isinstance(input, list):
            self._check_input_dim(input[0])
            # if evaluation, do it simple
            if not self.training:
                return my_data_parallel(self, input)
            if len(input) == 1:
                return self.forward(input[0])
            # calculate mean and var using multithreading
            all_sum, all_xsquare = {},{}
            def _worker(i, x, lock):
                try:
                    with torch.cuda.device_of(x):
                        xsum, xsquare = sum_square(x)
                    with lock:
                        all_sum[i] = xsum 
                        all_xsquare[i] = xsquare 
                except Exception as e:
                    with lock:
                        all_sum[i] = e
                        all_xsquare[i] = e
            threads = [threading.Thread(target=_worker,
                                        args=(i, x, self.writelock))
                        for i, x in enumerate(input)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            # convert to list
            def _to_list(x):
                outputs = []
                for i in range(len(x)):
                    outputs.append(x[i])
                return outputs
            
            all_sum = _to_list(all_sum)
            all_xsquare = _to_list(all_xsquare)
            xsums = AllReduce()(*all_sum)
            xsquares = AllReduce()(*all_xsquare)

            nGPUs = len(input)
            N = nGPUs * input[0].size(0)*input[0].size(2)*input[0].size(3)
            assert(N>1)
            xmean = xsums[0].data / N
            unbias_var = (xsquares[0].data - N * xmean * xmean) / (N-1) 
            # update running_mean and var
            self.running_mean = (1-self.momentum) * self.running_mean \
                + self.momentum * xmean
            self.running_var = (1-self.momentum) * self.running_var + \
                self.momentum * unbias_var
            # Broadcast the weight, bias, mean, std
            device_ids = list(range(torch.cuda.device_count()))
            weights = Broadcast(device_ids[:len(input)])(self.weight) 
            biases = Broadcast(device_ids[:len(input)])(self.bias)
            # parallel-apply
            results = {}
            def _worker_bn(i, x, xsum, xsquare, weight, bias, lock):
                var_input = _get_a_var(x)
                mean = xsum / N
                std  = (xsquare / N - mean * mean + self.eps).sqrt()
                try:
                    with torch.cuda.device_of(var_input):
                        B, C, H, W = x.size()
                        result = batchnormtrain(
                            x.view(B,C, -1), weight, bias, mean, 
                            std).view(B, C, H, W)
                    with lock: 
                        results[i] = result
                except Exception as e:
                    with lock:
                        results[i] = e
            threads = [threading.Thread(target=_worker_bn,
                                        args=(i, x, xsum, xsquare, weight, 
                                              bias, self.writelock)
                                       )
                        for i,( x, xsum, xsquare, weight, bias) in 
                        enumerate(zip(input, xsums, xsquares, 
                                      weights, biases))]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            outputs = []
            for i in range(len(results)):
                output = results[i]
                if isinstance(output, Exception):
                    raise output
                outputs.append(output)
            return outputs
        else:
            raise RuntimeError('unknown input type')


def _get_a_var(obj):
    if isinstance(obj, Variable):
        return obj

    if isinstance(obj, list) or isinstance(obj, tuple):
        results = map(_get_a_var, obj)
        for result in results:
            if isinstance(result, Variable):
                return result
    if isinstance(obj, dict):
        results = map(_get_a_var, obj.items())
        for result in results:
            if isinstance(result, Variable):
                return result
    return None
