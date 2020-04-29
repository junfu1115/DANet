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
from torch.nn.functional import interpolate

from .base import BaseNet
from .fcn import FCNHead

class DeepLabV3(BaseNet):
    r"""DeepLabV3

    Parameters
    ----------
    nclass : int
        Number of categories for the training dataset.
    backbone : string
        Pre-trained dilated backbone network type (default:'resnet50'; 'resnet50',
        'resnet101' or 'resnet152').
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.nn.BatchNorm`;
        for Synchronized Cross-GPU BachNormalization).
    aux : bool
        Auxiliary loss.


    Reference:

        Chen, Liang-Chieh, et al. "Rethinking atrous convolution for semantic image segmentation."
        arXiv preprint arXiv:1706.05587 (2017).

    """
    def __init__(self, nclass, backbone, aux=True, se_loss=False, norm_layer=nn.BatchNorm2d, **kwargs):
        super(DeepLabV3, self).__init__(nclass, backbone, aux, se_loss, norm_layer=norm_layer, **kwargs)
        self.head = DeepLabV3Head(2048, nclass, norm_layer, self._up_kwargs)
        if aux:
            self.auxlayer = FCNHead(1024, nclass, norm_layer)

    def forward(self, x):
        _, _, h, w = x.size()
        c1, c2, c3, c4 = self.base_forward(x)

        outputs = []
        x = self.head(c4)
        x = interpolate(x, (h,w), **self._up_kwargs)
        outputs.append(x)
        if self.aux:
            auxout = self.auxlayer(c3)
            auxout = interpolate(auxout, (h,w), **self._up_kwargs)
            outputs.append(auxout)
        return tuple(outputs)


class DeepLabV3Head(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, up_kwargs, atrous_rates=[12, 24, 36], **kwargs):
        super(DeepLabV3Head, self).__init__()
        inter_channels = in_channels // 8
        self.aspp = ASPP_Module(in_channels, atrous_rates, norm_layer, up_kwargs, **kwargs)
        self.block = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels),
            nn.ReLU(True),
            nn.Dropout(0.1, False),
            nn.Conv2d(inter_channels, out_channels, 1))

    def forward(self, x):
        x = self.aspp(x)
        x = self.block(x)
        return x


def ASPPConv(in_channels, out_channels, atrous_rate, norm_layer):
    block = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=atrous_rate,
                  dilation=atrous_rate, bias=False),
        norm_layer(out_channels),
        nn.ReLU(True))
    return block

class AsppPooling(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, up_kwargs):
        super(AsppPooling, self).__init__()
        self._up_kwargs = up_kwargs
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                 norm_layer(out_channels),
                                 nn.ReLU(True))

    def forward(self, x):
        _, _, h, w = x.size()
        pool = self.gap(x)
        return interpolate(pool, (h,w), **self._up_kwargs)

class ASPP_Module(nn.Module):
    def __init__(self, in_channels, atrous_rates, norm_layer, up_kwargs):
        super(ASPP_Module, self).__init__()
        out_channels = in_channels // 8
        rate1, rate2, rate3 = tuple(atrous_rates)
        self.b0 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(True))
        self.b1 = ASPPConv(in_channels, out_channels, rate1, norm_layer)
        self.b2 = ASPPConv(in_channels, out_channels, rate2, norm_layer)
        self.b3 = ASPPConv(in_channels, out_channels, rate3, norm_layer)
        self.b4 = AsppPooling(in_channels, out_channels, norm_layer, up_kwargs)

        self.project = nn.Sequential(
            nn.Conv2d(5*out_channels, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(True),
            nn.Dropout2d(0.5, False))

    def forward(self, x):
        feat0 = self.b0(x)
        feat1 = self.b1(x)
        feat2 = self.b2(x)
        feat3 = self.b3(x)
        feat4 = self.b4(x)
        y = torch.cat((feat0, feat1, feat2, feat3, feat4), 1)
        return self.project(y)

def get_deeplab(dataset='pascal_voc', backbone='resnet50s', pretrained=False,
            root='~/.encoding/models', **kwargs):
    # infer number of classes
    from ...datasets import datasets, acronyms
    model = DeepLabV3(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root, **kwargs)
    if pretrained:
        from ..model_store import get_model_file
        model.load_state_dict(torch.load(
            get_model_file('deeplab_%s_%s'%(backbone, acronyms[dataset]), root=root)))
    return model

def get_deeplab_resnet50_ade(pretrained=False, root='~/.encoding/models', **kwargs):
    r"""DeepLabV3 model from the paper `"Context Encoding for Semantic Segmentation"
    <https://arxiv.org/pdf/1803.08904.pdf>`_

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.encoding/models'
        Location for keeping the model parameters.


    Examples
    --------
    >>> model = get_deeplab_resnet50_ade(pretrained=True)
    >>> print(model)
    """
    return get_deeplab('ade20k', 'resnet50s', pretrained, root=root, **kwargs)

def get_deeplab_resnest50_ade(pretrained=False, root='~/.encoding/models', **kwargs):
    r"""DeepLabV3 model from the paper `"Context Encoding for Semantic Segmentation"
    <https://arxiv.org/pdf/1803.08904.pdf>`_

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.encoding/models'
        Location for keeping the model parameters.


    Examples
    --------
    >>> model = get_deeplab_resnet50_ade(pretrained=True)
    >>> print(model)
    """
    return get_deeplab('ade20k', 'resnest50', pretrained, aux=True, root=root, **kwargs)

def get_deeplab_resnest101_ade(pretrained=False, root='~/.encoding/models', **kwargs):
    r"""DeepLabV3 model from the paper `"Context Encoding for Semantic Segmentation"
    <https://arxiv.org/pdf/1803.08904.pdf>`_

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.encoding/models'
        Location for keeping the model parameters.


    Examples
    --------
    >>> model = get_deeplab_resnet50_ade(pretrained=True)
    >>> print(model)
    """
    return get_deeplab('ade20k', 'resnest101', pretrained, aux=True, root=root, **kwargs)

def get_deeplab_resnest200_ade(pretrained=False, root='~/.encoding/models', **kwargs):
    r"""DeepLabV3 model from the paper `"Context Encoding for Semantic Segmentation"
    <https://arxiv.org/pdf/1803.08904.pdf>`_

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.encoding/models'
        Location for keeping the model parameters.


    Examples
    --------
    >>> model = get_deeplab_resnest200_ade(pretrained=True)
    >>> print(model)
    """
    return get_deeplab('ade20k', 'resnest200', pretrained, aux=True, root=root, **kwargs)


def get_deeplab_resnest269_ade(pretrained=False, root='~/.encoding/models', **kwargs):
    r"""DeepLabV3 model from the paper `"Context Encoding for Semantic Segmentation"
    <https://arxiv.org/pdf/1803.08904.pdf>`_

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.encoding/models'
        Location for keeping the model parameters.


    Examples
    --------
    >>> model = get_deeplab_resnest200_ade(pretrained=True)
    >>> print(model)
    """
    return get_deeplab('ade20k', 'resnest269', pretrained, aux=True, root=root, **kwargs)

def get_deeplab_resnest50_pcontext(pretrained=False, root='~/.encoding/models', **kwargs):
    r"""DeepLabV3 model from the paper `"Context Encoding for Semantic Segmentation"
    <https://arxiv.org/pdf/1803.08904.pdf>`_

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.encoding/models'
        Location for keeping the model parameters.

    Examples
    --------
    >>> model = get_deeplab_resnest101_pcontext(pretrained=True)
    >>> print(model)
    """
    return get_deeplab('pcontext', 'resnest50', pretrained, aux=True, root=root, **kwargs)

def get_deeplab_resnest101_pcontext(pretrained=False, root='~/.encoding/models', **kwargs):
    r"""DeepLabV3 model from the paper `"Context Encoding for Semantic Segmentation"
    <https://arxiv.org/pdf/1803.08904.pdf>`_

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.encoding/models'
        Location for keeping the model parameters.


    Examples
    --------
    >>> model = get_deeplab_resnest101_pcontext(pretrained=True)
    >>> print(model)
    """
    return get_deeplab('pcontext', 'resnest101', pretrained, aux=True, root=root, **kwargs)

def get_deeplab_resnest200_pcontext(pretrained=False, root='~/.encoding/models', **kwargs):
    r"""DeepLabV3 model from the paper `"Context Encoding for Semantic Segmentation"
    <https://arxiv.org/pdf/1803.08904.pdf>`_

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.encoding/models'
        Location for keeping the model parameters.


    Examples
    --------
    >>> model = deeplab_resnest200_pcontext(pretrained=True)
    >>> print(model)
    """
    return get_deeplab('pcontext', 'resnest200', pretrained, aux=True, root=root, **kwargs)


def get_deeplab_resnest269_pcontext(pretrained=False, root='~/.encoding/models', **kwargs):
    r"""DeepLabV3 model from the paper `"Context Encoding for Semantic Segmentation"
    <https://arxiv.org/pdf/1803.08904.pdf>`_

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.encoding/models'
        Location for keeping the model parameters.


    Examples
    --------
    >>> model = get_deeplab_resnest269_pcontext(pretrained=True)
    >>> print(model)
    """
    return get_deeplab('pcontext', 'resnest269', pretrained, aux=True, root=root, **kwargs)
