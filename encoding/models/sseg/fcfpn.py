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

torch_ver = torch.__version__[:3]

__all__ = ['FCFPN', 'get_fcfpn', 'get_fcfpn_50_ade']

class FCFPN(BaseNet):
    r"""Fully Convolutional Networks for Semantic Segmentation

    Parameters
    ----------
    nclass : int
        Number of categories for the training dataset.
    backbone : string
        Pre-trained dilated backbone network type (default:'resnet50'; 'resnet50',
        'resnet101' or 'resnet152').
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.nn.BatchNorm`;


    Reference:

        Long, Jonathan, Evan Shelhamer, and Trevor Darrell. "Fully convolutional networks
        for semantic segmentation." *CVPR*, 2015

    Examples
    --------
    >>> model = FCFPN(nclass=21, backbone='resnet50')
    >>> print(model)
    """
    def __init__(self, nclass, backbone, aux=True, se_loss=False, norm_layer=nn.BatchNorm2d, **kwargs):
        super(FCFPN, self).__init__(nclass, backbone, aux, se_loss, dilated=False, norm_layer=norm_layer)
        self.head = FCFPNHead(nclass, norm_layer, up_kwargs=self._up_kwargs)
        assert not aux, "FCFPN does not support aux loss"

    def forward(self, x):
        imsize = x.size()[2:]
        features = self.base_forward(x)

        x = list(self.head(*features))
        x[0] = upsample(x[0], imsize, **self._up_kwargs)
        return tuple(x)


class FCFPNHead(nn.Module):
    def __init__(self, out_channels, norm_layer=None, fpn_inchannels=[256, 512, 1024, 2048],
                 fpn_dim=256, up_kwargs=None):
        super(FCFPNHead, self).__init__()
        # bilinear upsample options
        assert up_kwargs is not None
        self._up_kwargs = up_kwargs
        fpn_lateral = []
        for fpn_inchannel in fpn_inchannels[:-1]:
            fpn_lateral.append(nn.Sequential(
                nn.Conv2d(fpn_inchannel, fpn_dim, kernel_size=1, bias=False),
                norm_layer(fpn_dim),
                nn.ReLU(inplace=True),
            ))
        self.fpn_lateral = nn.ModuleList(fpn_lateral)
        fpn_out = []
        for _ in range(len(fpn_inchannels) - 1):
            fpn_out.append(nn.Sequential(
                nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, padding=1, bias=False),
                norm_layer(fpn_dim),
                nn.ReLU(inplace=True),
            ))
        self.fpn_out = nn.ModuleList(fpn_out)
        self.c4conv = nn.Sequential(nn.Conv2d(fpn_inchannels[-1], fpn_dim, 3, padding=1, bias=False),
                                    norm_layer(fpn_dim),
                                    nn.ReLU())
        inter_channels = len(fpn_inchannels) * fpn_dim
        self.conv5 = nn.Sequential(nn.Conv2d(inter_channels, 512, 3, padding=1, bias=False),
                                   norm_layer(512),
                                   nn.ReLU(),
                                   nn.Dropout(0.1, False),
                                   nn.Conv2d(512, out_channels, 1))

    def forward(self, *inputs):
        c4 = inputs[-1]
        #se_pred = False
        if hasattr(self, 'extramodule'):
            #if self.extramodule.se_loss:
            #    se_pred = True
            #    feat, se_out = self.extramodule(feat)
            #else:
            c4 = self.extramodule(c4)
        feat = self.c4conv(c4)
        c1_size = inputs[0].size()[2:]
        feat_up = upsample(feat, c1_size, **self._up_kwargs)
        fpn_features = [feat_up]
        # c4, c3, c2, c1
        for i in reversed(range(len(inputs) - 1)):
            feat_i = self.fpn_lateral[i](inputs[i])
            feat = upsample(feat, feat_i.size()[2:], **self._up_kwargs)
            feat = feat + feat_i
            # upsample to the same size with c1
            feat_up = upsample(self.fpn_out[i](feat), c1_size, **self._up_kwargs)
            fpn_features.append(feat_up)
        fpn_features = torch.cat(fpn_features, 1)
        #if se_pred:
        #    return (self.conv5(fpn_features), se_out)
        return (self.conv5(fpn_features), )


def get_fcfpn(dataset='pascal_voc', backbone='resnet50', pretrained=False,
            root='~/.encoding/models', **kwargs):
    r"""FCFPN model from the paper `"Fully Convolutional Network for semantic segmentation"
    <https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcfpn.pdf>`_
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
    >>> model = get_fcfpn(dataset='pascal_voc', backbone='resnet50s', pretrained=False)
    >>> print(model)
    """
    acronyms = {
        'pascal_voc': 'voc',
        'pascal_aug': 'voc',
        'ade20k': 'ade',
    }
    # infer number of classes
    from ...datasets import datasets, VOCSegmentation, VOCAugSegmentation, ADE20KSegmentation
    model = FCFPN(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, **kwargs)
    if pretrained:
        from ..model_store import get_model_file
        model.load_state_dict(torch.load(
            get_model_file('fcfpn_%s_%s'%(backbone, acronyms[dataset]), root=root)))
    return model


def get_fcfpn_50_ade(pretrained=False, root='~/.encoding/models', **kwargs):
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
    >>> model = get_fcfpn_50_ade(pretrained=True)
    >>> print(model)
    """
    return get_fcfpn('ade20k', 'resnet50s', pretrained)

