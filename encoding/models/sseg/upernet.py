###########################################################################
# Created by: Hang Zhang 
# Email: zhang.hang@rutgers.edu 
# Copyright (c) 2017
###########################################################################
from __future__ import division
import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import upsample

from .base import BaseNet
from .fcfpn import FCFPNHead
from ...nn import PyramidPooling

torch_ver = torch.__version__[:3]

__all__ = ['UperNet', 'get_upernet', 'get_upernet_50_ade']

class UperNet(BaseNet):
    r"""Fully Convolutional Networks for Semantic Segmentation

    Parameters
    ----------
    nclass : int
        Number of categories for the training dataset.
    backbone : string
        Pre-trained dilated backbone network type (default:'resnet50s'; 'resnet50s',
        'resnet101s' or 'resnet152s').
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.nn.BatchNorm`;


    Reference:

        Long, Jonathan, Evan Shelhamer, and Trevor Darrell. "Fully convolutional networks
        for semantic segmentation." *CVPR*, 2015

    Examples
    --------
    >>> model = UperNet(nclass=21, backbone='resnet50s')
    >>> print(model)
    """
    def __init__(self, nclass, backbone, aux=True, se_loss=False, norm_layer=nn.BatchNorm2d, **kwargs):
        super(UperNet, self).__init__(nclass, backbone, aux, se_loss, dilated=False, norm_layer=norm_layer)
        self.head = UperNetHead(nclass, norm_layer, up_kwargs=self._up_kwargs)
        assert not aux, "UperNet does not support aux loss"

    def forward(self, x):
        imsize = x.size()[2:]
        features = self.base_forward(x)

        x = list(self.head(*features))
        x[0] = upsample(x[0], imsize, **self._up_kwargs)
        return tuple(x)


class UperNetHead(FCFPNHead):
    def __init__(self, out_channels, norm_layer=None, fpn_inchannels=[256, 512, 1024, 2048],
                 fpn_dim=256, up_kwargs=None):
        fpn_inchannels[-1] = fpn_inchannels[-1] * 2
        super(UperNetHead, self).__init__(out_channels, norm_layer, fpn_inchannels,
                                          fpn_dim, up_kwargs)
        self.extramodule = PyramidPooling(fpn_inchannels[-1] // 2, norm_layer, up_kwargs)


def get_upernet(dataset='pascal_voc', backbone='resnet50s', pretrained=False,
            root='~/.encoding/models', **kwargs):
    r"""UperNet model from the paper `"Fully Convolutional Network for semantic segmentation"
    <https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_upernet.pdf>`_
    Parameters
    ----------
    dataset : str, default pascal_voc
        The dataset that model pretrained on. (pascal_voc, ade20k)
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.encoding/models'
        Location for keeping the model parameters.
    Examples
    --------
    >>> model = get_upernet(dataset='pascal_voc', backbone='resnet50s', pretrained=False)
    >>> print(model)
    """
    acronyms = {
        'pascal_voc': 'voc',
        'pascal_aug': 'voc',
        'ade20k': 'ade',
    }
    # infer number of classes
    from ...datasets import datasets, VOCSegmentation, VOCAugSegmentation, ADE20KSegmentation
    model = UperNet(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, **kwargs)
    if pretrained:
        from ..model_store import get_model_file
        model.load_state_dict(torch.load(
            get_model_file('upernet_%s_%s'%(backbone, acronyms[dataset]), root=root)))
    return model


def get_upernet_50_ade(pretrained=False, root='~/.encoding/models', **kwargs):
    r"""EncNet-PSP model from the paper `"Context Encoding for Semantic Segmentation"
    <https://arxiv.org/pdf/1803.08904.pdf>`_

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.encoding/models'
        Location for keeping the model parameters.


    Examples
    --------
    >>> model = get_upernet_50_ade(pretrained=True)
    >>> print(model)
    """
    return get_upernet('ade20k', 'resnet50s', pretrained)
