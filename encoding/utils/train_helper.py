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

def get_selabel_vector(target, nclass):
    """Get SE-Loss Label in a batch
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

