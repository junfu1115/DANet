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
import collections
import threading
import torch
from torch.nn import Module, Sequential, Conv1d, Conv2d, ConvTranspose2d, \
    ReLU, Sigmoid, MaxPool2d, AvgPool2d, AdaptiveAvgPool2d, Dropout2d, Linear, \
    DataParallel
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.functional import batch_norm
from torch.nn.parallel._functions import ReduceAddCoalesced, Broadcast

from ..functions import *
from ..parallel import allreduce
from .comm import SyncMaster


__all__ = ['BatchNorm1d', 'BatchNorm2d', 'BatchNorm3d', 'Module', 'Sequential', 'Conv1d',
           'Conv2d', 'ConvTranspose2d', 'ReLU', 'Sigmoid', 'MaxPool2d', 'AvgPool2d',
           'AdaptiveAvgPool2d', 'Dropout2d', 'Linear']

class _SyncBatchNorm(_BatchNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super(_SyncBatchNorm, self).__init__(num_features, eps=eps, momentum=momentum, affine=affine)

        self._sync_master = SyncMaster(self._data_parallel_master)
        self._parallel_id = None
        self._slave_pipe = None

    def forward(self, input):
        if not self.training:
            return batch_norm(
                input, self.running_mean, self.running_var, self.weight, self.bias,
                self.training, self.momentum, self.eps)

        # Resize the input to (B, C, -1).
        input_shape = input.size()
        input = input.view(input_shape[0], self.num_features, -1)

        # sum(x) and sum(x^2)
        N = input.size(0) * input.size(2)
        xsum, xsqsum = sum_square(input)

        # all-reduce for global sum(x) and sum(x^2)
        if self._parallel_id == 0:
            mean, inv_std = self._sync_master.run_master(_ChildMessage(xsum, xsqsum, N))
        else:
            mean, inv_std = self._slave_pipe.run_slave(_ChildMessage(xsum, xsqsum, N))
        # forward
        return batchnormtrain(input, mean, 1.0/inv_std, self.weight, self.bias).view(input_shape)

    def __data_parallel_replicate__(self, ctx, copy_id):
        self._parallel_id = copy_id

        # parallel_id == 0 means master device.
        if self._parallel_id == 0:
            ctx.sync_master = self._sync_master
        else:
            self._slave_pipe = ctx.sync_master.register_slave(copy_id)

    def _data_parallel_master(self, intermediates):
        """Reduce the sum and square-sum, compute the statistics, and broadcast it."""

        # Always using same "device order" makes the ReduceAdd operation faster.
        # Thanks to:: Tete Xiao (http://tetexiao.com/)
        intermediates = sorted(intermediates, key=lambda i: i[1].sum.get_device())

        to_reduce = [i[1][:2] for i in intermediates]
        to_reduce = [j for i in to_reduce for j in i]  # flatten
        target_gpus = [i[1].sum.get_device() for i in intermediates]

        sum_size = sum([i[1].sum_size for i in intermediates])
        sum_, ssum = ReduceAddCoalesced.apply(target_gpus[0], 2, *to_reduce)
        mean, inv_std = self._compute_mean_std(sum_, ssum, sum_size)

        broadcasted = Broadcast.apply(target_gpus, mean, inv_std)

        outputs = []
        for i, rec in enumerate(intermediates):
            outputs.append((rec[0], _MasterMessage(*broadcasted[i*2:i*2+2])))

        return outputs

    def _compute_mean_std(self, sum_, ssum, size):
        """Compute the mean and standard-deviation with sum and square-sum. This method
        also maintains the moving average on the master device."""
        assert size > 1, 'BatchNorm computes unbiased standard-deviation, which requires size > 1.'
        mean = sum_ / size
        sumvar = ssum - sum_ * mean
        unbias_var = sumvar / (size - 1)
        bias_var = sumvar / size

        self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.data
        self.running_var = (1 - self.momentum) * self.running_var + self.momentum * unbias_var.data

        return mean, (bias_var + self.eps) ** -0.5


# API adapted from https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
_ChildMessage = collections.namedtuple('Message', ['sum', 'ssum', 'sum_size'])
_MasterMessage = collections.namedtuple('_MasterMessage', ['sum', 'inv_std'])


class BatchNorm1d(_SyncBatchNorm):
    r"""Please see the docs in :class:`encoding.nn.BatchNorm2d`"""
    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'
                             .format(input.dim()))
        super(BatchNorm2d, self)._check_input_dim(input)


class BatchNorm2d(_SyncBatchNorm):
    r"""Cross-GPU Synchronized Batch normalization (SyncBN)

    Standard BN [1]_ implementation only normalize the data within each device (GPU).
    SyncBN normalizes the input within the whole mini-batch.
    We follow the sync-onece implmentation described in the paper [2]_ .
    Please see the design idea in the `notes <./notes/syncbn.html>`_.

    .. note::
        We adapt the awesome python API from another `PyTorch SyncBN Implementation
        <https://github.com/vacancy/Synchronized-BatchNorm-PyTorch>`_ and provide
        efficient CUDA backend.

    .. math::

        y = \frac{x - mean[x]}{ \sqrt{Var[x] + \epsilon}} * gamma + beta

    The mean and standard-deviation are calculated per-channel over
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
        >>> m = BatchNorm2d(100)
        >>> net = torch.nn.DataParallel(m)
        >>> encoding.parallel.patch_replication_callback(net)
        >>> output = net(input)
    """
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))
        super(BatchNorm2d, self)._check_input_dim(input)


class BatchNorm3d(_SyncBatchNorm):
    r"""Please see the docs in :class:`encoding.nn.BatchNorm2d`"""
    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'
                             .format(input.dim()))
        super(BatchNorm3d, self)._check_input_dim(input)


class SharedTensor(object):
    """Shared Tensor for cross GPU all reduce operation"""
    def __init__(self, nGPUs):
        self.mutex = threading.Lock()
        self.all_tasks_done = threading.Condition(self.mutex)
        self.nGPUs = nGPUs
        self._clear()

    def _clear(self):
        self.N = 0
        self.dict = {}
        self.push_tasks = self.nGPUs
        self.reduce_tasks = self.nGPUs

    def push(self, *inputs):
        # push from device
        with self.mutex:
            if self.push_tasks == 0:
                self._clear()
            self.N += inputs[0]
            igpu = inputs[1]
            self.dict[igpu] = inputs[2:]
            #idx = self.nGPUs - self.push_tasks
            self.push_tasks -= 1
        with self.all_tasks_done:
            if self.push_tasks == 0:
                self.all_tasks_done.notify_all()
            while self.push_tasks:
                self.all_tasks_done.wait()

    def pull(self, igpu):
        # pull from device
        with self.mutex:
            if igpu == 0:
                assert(len(self.dict) == self.nGPUs)
                # flatten the tensors
                self.list = [t for i in range(len(self.dict)) for t in self.dict[i]]
                self.outlist = allreduce(2, *self.list)
                self.reduce_tasks -= 1
            else:
                self.reduce_tasks -= 1
        with self.all_tasks_done:
            if self.reduce_tasks == 0:
                self.all_tasks_done.notify_all()
            while self.reduce_tasks:
                self.all_tasks_done.wait()
        # all reduce done
        return self.N, self.outlist[2*igpu], self.outlist[2*igpu+1]

    def __len__(self):
        return self.nGPUs

    def __repr__(self):
        return ('SharedTensor')

