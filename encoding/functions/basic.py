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
import torch.nn.functional as F
from torch.autograd import Function, Variable

__all__ = ['squeeze_each', 'view_each', 'multi_each', 'sum_each', 
    'cat_each', 'upsample', 'dropout', 'relu']

def squeeze_each(x, dim=None):
    """Multi-GPU version torch. squeeze()
    
    """
    y = []
    for i in range(len(x)):
        if dim is None:
            y.append(x[i].squeeze())
        else:
            y.append(x[i].squeeze(dim))
    return y

def view_each(x, size):
    """Multi-GPU version torch.view

    Returns a new tensor with the same data but different size.
    The returned tensor shares the same data and must have the same number
    of elements, but may have a different size. A tensor must be
    :attr:`contiguous` to be viewed.

    Args:
        input: list of multi-gpu tensors
        size (torch.Size or int...): Desired size

    """
    y = []
    for i in range(len(x)):
        y.append(x[i].view(size))
    return y

def multi_each(a, b):
    """Multi-GPU version multiplication

    .. math::
        y[i] = a[i] * b[i]
    """
    y = []
    for i in range(len(a)):
        y.append(a[i] * b[i])
    return y

def sum_each(x, y):
    """Multi-GPU version torch.add

    .. math::
        y[i] = a[i] + b[i]
    """
    assert(len(x)==len(y))
    z = []
    for i in range(len(x)):
        z.append(x[i]+y[i])
    return z


def cat_each(x1, x2, dim):
    """Multi-GPU version torch.cat

    .. math::
        y[i] = torch.cat(a[i], b[i], dim)
    """
    assert(len(x1)==len(x2))
    z = []
    for i in range(len(x1)):
        with torch.cuda.device_of(x1[i]):
            x = torch.cat((x1[i], x2[i]), dim)
            z.append(x)
    return z


def dict_to_list(x):
    """Converting Dict{} to list[]
    """
    y = []
    for i in range(len(x)):
        xi = x[i]
        if isinstance(xi, Exception):
            raise xi
        y.append(xi)
    return y


def upsample(input, size=None, scale_factor=None, mode='nearest'):
    """Multi-GPU version torch.nn.functional.upsample

    Upsamples the input to either the given :attr:`size` or the given
    :attr:`scale_factor`

    The algorithm used for upsampling is determined by :attr:`mode`.

    Currently temporal, spatial and volumetric upsampling are supported, i.e.
    expected inputs are 3-D, 4-D or 5-D in shape.

    The input dimensions are interpreted in the form:
    `mini-batch x channels x [depth] x [height] x width`

    The modes available for upsampling are: `nearest`, `linear` (3D-only),
    `bilinear` (4D-only), `trilinear` (5D-only)

    Args:
        input (Variable): input
        size (int or Tuple[int] or Tuple[int, int] or Tuple[int, int, int]):
            output spatial size.
        scale_factor (int): multiplier for spatial size. Has to be an integer.
        mode (string): algorithm used for upsampling:
            'nearest' | 'linear' | 'bilinear' | 'trilinear'. Default: 'nearest'
    """
    if isinstance(input, Variable):
        return F.upsample(input, size=size, scale_factor=scale_factor,
                          mode=mode)
    elif isinstance(input, tuple) or isinstance(input, list):
        lock = threading.Lock()
        results = {}
        def _worker(i, x):
            try:
                with torch.cuda.device_of(x):
                    result =  F.upsample(x, size=size, \
                        scale_factor=scale_factor,mode=mode)
                with lock:
                    results[i] = result
            except Exception as e:
                with lock:
                    resutls[i] = e 
        # multi-threading for different gpu
        threads = [threading.Thread(target=_worker,
                                    args=(i, x),
                                    )
                   for i, (x) in enumerate(input)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join() 
        outputs = dict_to_list(results)
        return outputs
    else:
        raise RuntimeError('unknown input type')


def dropout(input, p=0.5, training=False, inplace=True):
    """Multi-GPU version torch.nn.functional.droupout

    The channels to zero-out are randomized on every forward call.

    *Usually the input comes from Conv2d modules.*

    As described in the paper
    `Efficient Object Localization Using Convolutional Networks`,
    if adjacent pixels within feature maps are strongly correlated
    (as is normally the case in early convolution layers) then iid dropout
    will not regularize the activations and will otherwise just result
    in an effective learning rate decrease.

    In this case, :func:`nn.Dropout2d` will help promote independence between
    feature maps and should be used instead.

    Args:
        p (float, optional): probability of an element to be zeroed.
        inplace (bool, optional): If set to True, will do this operation
            in-place

    Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)` (same shape as input)
    
    """
    if isinstance(input, Variable):
        return F.dropout(input, p, training, inplace)
    elif isinstance(input, tuple) or isinstance(input, list):
        lock = threading.Lock()
        results = {}
        def _worker(i, x):
            try:
                with torch.cuda.device_of(x):
                    result =  F.dropout(x, p, training, inplace)
                with lock:
                    results[i] = result
            except Exception as e:
                with lock:
                    resutls[i] = e 
        # multi-threading for different gpu
        threads = [threading.Thread(target=_worker,
                                    args=(i, x),
                                    )
                   for i, (x) in enumerate(input)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join() 
        outputs = dict_to_list(results)
        return outputs
    else:
        raise RuntimeError('unknown input type')

def relu(input, inplace=False):
    """Multi-GPU version torch.nn.functional.relu

    Applies the rectified linear unit function element-wise
    :math:`{ReLU}(x)= max(0, x)`

    Args:
        inplace: can optionally do the operation in-place. Default: False

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input
    """
    if isinstance(input, Variable):
        return F.relu(input, inplace)
    elif isinstance(input, tuple) or isinstance(input, list):
        lock = threading.Lock()
        results = {}
        def _worker(i, x):
            try:
                with torch.cuda.device_of(x):
                    result =  F.relu(x, inplace)
                with lock:
                    results[i] = result
            except Exception as e:
                with lock:
                    resutls[i] = e 
        # multi-threading for different gpu
        threads = [threading.Thread(target=_worker,
                                    args=(i, x),
                                    )
                   for i, (x) in enumerate(input)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join() 
        outputs = dict_to_list(results)
        return outputs

    else:
        raise RuntimeError('unknown input type')
