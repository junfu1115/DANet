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
import torch.cuda.nccl as nccl
import torch.cuda.comm as comm
from torch.autograd import Variable, Function
from torch.nn.modules import Module
from torch.nn.parallel.scatter_gather import scatter, scatter_kwargs, \
    gather
from torch.nn.parallel.replicate import replicate
from torch.nn.parallel.parallel_apply import parallel_apply

__all__ = ['AllReduce', 'Broadcast', 'ModelDataParallel', 
    'CriterionDataParallel', 'SelfDataParallel']

def nccl_all_reduce(inputs):
    # TODO, figure out why nccl all_reduce doesn't work for gradcheck
    input_size = inputs[0].size()
    #if nccl.is_available(inputs):
    for i, inp in enumerate(inputs):
        assert inp.is_cuda, \
            "reduce_add expects all inputs to be on GPUs"
        if inp.size() != input_size:
            got = 'x'.join(str(x) for x in inp.size())
            expected = 'x'.join(str(x) for x in input_size)
            raise ValueError("input {} has invalid size: got {}, \
                but expected {}".format(i, got, expected))
    nccl.all_reduce(inputs)
    return inputs

def comm_all_reduce(inputs):
    # comm backend
    result = comm.reduce_add(inputs)
    results = []
    for i in range(len(inputs)):
        results.append(result.clone().cuda(i))
    return results


class AllReduce(Function):
    """Cross GPU all reduce autograd operation for calculate mean and
    variance in SyncBN.
    """
    def forward(ctx, *inputs):
        outputs = comm_all_reduce(list(inputs))
        return tuple(outputs)

    def backward(ctx, *gradOutputs):
        gradInputs = comm_all_reduce(list(gradOutputs))
        return tuple(gradInputs)


class Broadcast(Function):
    """Multi-GPU broadcast autograd function
    """
    def __init__(self, target_gpus):
        super(Broadcast, self).__init__()
        self.target_gpus = target_gpus

    def forward(self, *inputs):
        if not all(input.is_cuda for input in inputs):
            raise TypeError('Broadcast function not implemented for CPU tensors')
        if len(inputs) == 0:
            return tuple()
        self.num_inputs = len(inputs)
        self.input_device = inputs[0].get_device()
        outputs = comm.broadcast_coalesced(inputs, self.target_gpus)
        return tuple([t for tensors in outputs for t in tensors])

    def backward(self, *grad_outputs):
        grad_outputs = [grad_outputs[i:i + self.num_inputs]
                        for i in range(0, len(grad_outputs), self.num_inputs)]
        return comm.reduce_add_coalesced(grad_outputs, self.input_device)


class ModelDataParallel(Module):
    """Implements data parallelism at the module level.

    Reference::
        We provide this code for a comming paper.

    This container parallelizes the application of the given module by
    splitting the input across the specified devices by chunking in the 
    batch dimension. 
    In the forward pass, the module is replicated on each device,
    and each replica handles a portion of the input. During the backwards
    pass, gradients from each replica are summed into the original module.
    Note that the outputs are not gathered, please use compatible 
    :class:`encoding.parallel.CriterionDataParallel`.

    The batch size should be larger than the number of GPUs used. It should
    also be an integer multiple of the number of GPUs so that each chunk is
    the same size (so that each GPU processes the same number of samples).

    Args:
        module: module to be parallelized
        device_ids: CUDA devices (default: all devices)

    Example::

        >>> net = encoding.nn.ModelDataParallel(model, device_ids=[0, 1, 2])
        >>> output = net(input_var)
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
        """
        # TODO FIXME temporal solution for BN
        for m in self.module.modules():
            classname = m.__class__.__name__ 
            if classname.find('BatchNorm2d') != -1:
                m.momentum = 0.9996
        """

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

    Reference::
        We provide this code for a comming paper.

    The targets are splitted across the specified devices by chunking in
    the batch dimension. Please use together with :class:`encoding.parallel.ModelDataParallel`.
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
        return self.gather(outputs, self.output_device)

    def replicate(self, module, device_ids):
        return replicate(module, device_ids)

    def scatter(self, inputs, kwargs, device_ids):
        return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)

    def parallel_apply(self, replicas, inputs, targets, kwargs):
        return criterion_parallel_apply(replicas, inputs, targets, kwargs)

    def gather(self, outputs, output_device):
        return gather(outputs, output_device, dim=self.dim).mean()
    

class SelfDataParallel(Module):
    """SelfDataParallel, please make sure you understand it before using.

    Reference::
        We provide this code for a comming paper.

    Each module in the network should be in self-parallel mode, 
    which allows list of inputs from multiple GPUs.
    Please see :class:`encoding.nn` for detail, use with cautious
    """
    def __init__(self, module, device_ids=None, output_device=None, dim=0):
        super(SelfDataParallel, self).__init__()
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
        if self.training:
            # self parallel mode
            outputs = self.module(inputs)
            return outputs
        else:
            # TODO check faster?
            if len(self.device_ids) == 1:
                return self.module(*inputs[0], **kwargs[0])
            replicas = self.replicate(self.module, \
                self.device_ids[:len(inputs)])
            outputs = self.parallel_apply(replicas, inputs, kwargs)
            return outputs 
            
    def replicate(self, module, device_ids):
        return replicate(module, device_ids)

    def parallel_apply(self, replicas, inputs, kwargs):
        return parallel_apply(replicas, inputs, kwargs)

    def scatter(self, inputs, kwargs, device_ids):
        outputs = scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)
        return outputs


def criterion_parallel_apply(modules, inputs, targets, kwargs_tup=None):
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
        var_input = input
        while not isinstance(var_input, Variable):
            var_input = var_input[0]
        var_target = target
        while not isinstance(var_target, Variable):
            var_target = var_target[0]
        try:
            with torch.cuda.device_of(var_input):
                output = module(input, *target, **kwargs)
            with lock:
                results[i] = output
        except Exception as e:
            with lock:
                results[i] = e

    threads = [threading.Thread(target=_worker,
                                args=(i, module, input, target, 
                                      kwargs, results, lock),
                                )
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


def get_a_var(obj):
    if isinstance(obj, Variable):
        return obj

    if isinstance(obj, list) or isinstance(obj, tuple):
        results = map(get_a_var, obj)
        for result in results:
            if isinstance(result, Variable):
                return result
    if isinstance(obj, dict):
        results = map(get_a_var, obj.items())
        for result in results:
            if isinstance(result, Variable):
                return result
    return None


def my_parallel_apply(modules, inputs, kwargs_tup=None):
    assert len(modules) == len(inputs)
    if kwargs_tup:
        assert len(modules) == len(kwargs_tup)
    else:
        kwargs_tup = ({},) * len(modules)
    # Fast track
    if len(modules) == 1:
        return (modules[0](*inputs[0], **kwargs_tup[0]), )

    lock = threading.Lock()
    results = {}

    def _worker(i, module, input, kwargs, results, lock):
        var_input = get_a_var(input)
        try:
            with torch.cuda.device_of(var_input):
                output = module(input, **kwargs)
            with lock:
                results[i] = output
        except Exception as e:
            with lock:
                results[i] = e

    threads = [threading.Thread(target=_worker,
                                args=(i, module, input, kwargs, results, lock),
                                )
               for i, (module, input, kwargs) in
               enumerate(zip(modules, inputs, kwargs_tup))]

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


def my_data_parallel(module, inputs, device_ids=None, \
    dim=0, module_kwargs=None):
    if device_ids is None:
        device_ids = list(range(torch.cuda.device_count()))

    if len(inputs) == 1:
        return module(inputs[0])

    #print('my data parallel, len(inputs)', len(inputs))
    replicas = replicate(module, device_ids[:len(inputs)])
    outputs = my_parallel_apply(replicas, inputs, module_kwargs)
    return outputs 


