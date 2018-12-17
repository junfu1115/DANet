##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## ECE Department, Rutgers University
## Email: zhang.hang@rutgers.edu
## Copyright (c) 2017
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import torch
import torch.nn as nn

#from ..nn import SyncBatchNorm
from torch.nn.modules.batchnorm import _BatchNorm

__all__ = ['get_selabel_vector']

def get_selabel_vector(target, nclass):
    r"""Get SE-Loss Label in a batch
    Args:
        predict: input 4D tensor
        target: label 3D tensor (BxHxW)
        nclass: number of categories (int)
    Output:
        2D tensor (BxnClass)
    """
    batch = target.size(0)
    tvect = torch.zeros(batch, nclass)
    for i in range(batch):
        hist = torch.histc(target[i].data.float(), 
                           bins=nclass, min=0,
                           max=nclass-1)
        vect = hist>0
        tvect[i] = vect
    return tvect


class EMA():
    r""" Use moving avg for the models.
    Examples:
        >>> ema = EMA(0.999)
        >>> for name, param in model.named_parameters():
        >>>     if param.requires_grad:
        >>>         ema.register(name, param.data)
        >>> 
        >>> # during training:
        >>> # optimizer.step()
        >>> for name, param in model.named_parameters():
        >>>    # Sometime I also use the moving average of non-trainable parameters, just according to the model structure
        >>>    if param.requires_grad:
        >>>         ema(name, param.data)
        >>> 
        >>> # during eval or test
        >>> import copy
        >>> model_test = copy.deepcopy(model)
        >>> for name, param in model_test.named_parameters():
        >>>    # Sometime I also use the moving average of non-trainable parameters, just according to the model structure
        >>>    if param.requires_grad:
        >>>         param.data = ema.get(name)
        >>> # Then use model_test for eval.
    """
    def __init__(self, momentum):
        self.momentum = momentum
        self.shadow = {}

    def register(self, name, val):
        self.shadow[name] = val.clone()

    def __call__(self, name, x):
        assert name in self.shadow
        new_average = (1.0 - self.momentum) * x + self.momentum * self.shadow[name]
        self.shadow[name] = new_average.clone()
        return new_average

    def get(self, name):
        assert name in self.shadow
        return self.shadow[name]
