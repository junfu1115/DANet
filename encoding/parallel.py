##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## ECE Department, Rutgers University
## Email: zhang.hang@rutgers.edu
## Copyright (c) 2017
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

"""Encoding Data Parallel"""
import threading
import torch
from torch.autograd import Variable
from torch.nn.modules import Module
from torch.nn.parallel.scatter_gather import scatter_kwargs
from torch.nn.parallel.replicate import replicate
from torch.nn.parallel.parallel_apply import parallel_apply
from torch.nn.parallel._functions import ReduceAddCoalesced, Broadcast

__all__ = ['allreduce', 'ModelDataParallel', 'CriterionDataParallel']


def allreduce(*inputs):
    """Cross GPU all reduce autograd operation for calculate mean and
    variance in SyncBN.
    """
    target_gpus = [inputs[i].get_device() for i in range(len(inputs))]
    result = ReduceAddCoalesced.apply(target_gpus[0], 1, *inputs)
    outputs = Broadcast.apply(target_gpus, *result)
    assert len(outputs) == len(inputs)
    return outputs


class ModelDataParallel(Module):
    """Implements data parallelism at the module level.

    This container parallelizes the application of the given module by
    splitting the input across the specified devices by chunking in the
    batch dimension.
    In the forward pass, the module is replicated on each device,
    and each replica handles a portion of the input. During the backwards pass, gradients from each replica are summed into the original module.
    Note that the outputs are not gathered, please use compatible
    :class:`encoding.parallel.CriterionDataParallel`.

    The batch size should be larger than the number of GPUs used. It should
    also be an integer multiple of the number of GPUs so that each chunk is
    the same size (so that each GPU processes the same number of samples).

    Args:
        module: module to be parallelized
        device_ids: CUDA devices (default: all devices)

    Reference:
        Hang Zhang, Kristin Dana, Jianping Shi, Zhongyue Zhang, Xiaogang Wang, Ambrish Tyagi,
        Amit Agrawal. “Context Encoding for Semantic Segmentation.
        *The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) 2018*

    Example::

        >>> net = encoding.nn.ModelDataParallel(model, device_ids=[0, 1, 2])
        >>> y = net(x)
    """
    def __init__(self, module, device_ids=None, output_device=None, dim=0):
        super(ModelDataParallel, self).__init__()
        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))
        if output_device is None:
            output_device = device_ids[0]
        self.dim = dim
        self.module = module
        self.device_ids = device_ids
        self.output_device = output_device
        self.master_mean, self.master_var = {}, {}
        if len(self.device_ids) == 1:
            self.module.cuda(device_ids[0])

    def forward(self, *inputs, **kwargs):
        inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
        if len(self.device_ids) == 1:
            return self.module(*inputs[0], **kwargs[0])
        replicas = self.replicate(self.module, \
            self.device_ids[:len(inputs)])
        outputs = self.parallel_apply(replicas, inputs, kwargs)
        return outputs

    def replicate(self, module, device_ids):
        return replicate(module, device_ids)

    def scatter(self, inputs, kwargs, device_ids):
        return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)

    def parallel_apply(self, replicas, inputs, kwargs):
        return parallel_apply(replicas, inputs, kwargs)


class CriterionDataParallel(Module):
    """
    Calculate loss in multiple-GPUs, which balance the memory usage for
    Semantic Segmentation.

    The targets are splitted across the specified devices by chunking in
    the batch dimension. Please use together with :class:`encoding.parallel.ModelDataParallel`.

    Reference:
        Hang Zhang, Kristin Dana, Jianping Shi, Zhongyue Zhang, Xiaogang Wang, Ambrish Tyagi,
        Amit Agrawal. “Context Encoding for Semantic Segmentation.
        *The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) 2018*

    Example::

        >>> net = encoding.nn.ModelDataParallel(model, device_ids=[0, 1, 2])
        >>> criterion = encoding.nn.CriterionDataParallel(criterion, device_ids=[0, 1, 2])
        >>> y = net(x)
        >>> loss = criterion(y, target)
    """
    def __init__(self, module, device_ids=None, output_device=None, dim=0):
        super(CriterionDataParallel, self).__init__()
        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))
        if output_device is None:
            output_device = device_ids[0]
        self.dim = dim
        self.module = module
        self.device_ids = device_ids
        self.output_device = output_device
        if len(self.device_ids) == 1:
            self.module.cuda(device_ids[0])

    def forward(self, inputs, *targets, **kwargs):
        # input should be already scatterd
        # scattering the targets instead
        targets, kwargs = self.scatter(targets, kwargs, self.device_ids)
        if len(self.device_ids) == 1:
            return self.module(inputs, *targets[0], **kwargs[0])
        replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
        outputs = self.parallel_apply(replicas, inputs, targets, kwargs)
        return ReduceAddCoalesced.apply(self.output_device, 1, *outputs) / len(outputs)

    def replicate(self, module, device_ids):
        return replicate(module, device_ids)

    def scatter(self, inputs, kwargs, device_ids):
        return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)

    def parallel_apply(self, replicas, inputs, targets, kwargs):
        return _criterion_parallel_apply(replicas, inputs, targets, kwargs)


def _criterion_parallel_apply(modules, inputs, targets, kwargs_tup=None):
    assert len(modules) == len(inputs)
    assert len(targets) == len(inputs)
    if kwargs_tup:
        assert len(modules) == len(kwargs_tup)
    else:
        kwargs_tup = ({},) * len(modules)
    # Fast track
    if len(modules) == 1:
        return (modules[0](*inputs[0], *targets[0], **kwargs_tup[0]), )

    lock = threading.Lock()
    results = {}

    def _worker(i, module, input, target, kwargs, results, lock):
        try:
            var_input = _get_a_var(input)
            with torch.cuda.device_of(var_input):
                output = module(input, *target, **kwargs)
            with lock:
                results[i] = output
        except Exception as e:
            with lock:
                results[i] = e

    threads = [threading.Thread(target=_worker,
                                args=(i, module, input, target,
                                      kwargs, results, lock),)
               for i, (module, input, target, kwargs) in
               enumerate(zip(modules, inputs, targets, kwargs_tup))]

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    outputs = []
    for i in range(len(inputs)):
        output = results[i]
        if isinstance(output, Exception):
            raise output
        outputs.append(output)
    return outputs


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
