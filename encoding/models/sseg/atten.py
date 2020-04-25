###########################################################################
# Created by: Hang Zhang 
# Email: zhang.hang@rutgers.edu 
# Copyright (c) 2018
###########################################################################
from __future__ import division
import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import interpolate

from .base import BaseNet
from ...nn import ACFModule, ConcurrentModule, SyncBatchNorm
from .fcn import FCNHead
from .encnet import EncModule

__all__ = ['ATTEN', 'get_atten']

class ATTEN(BaseNet):
    def __init__(self, nclass, backbone, nheads=8, nmixs=1, with_global=True,
                 with_enc=True, with_lateral=False, aux=True, se_loss=False,
                 norm_layer=SyncBatchNorm, **kwargs):
        super(ATTEN, self).__init__(nclass, backbone, aux, se_loss,
                                    norm_layer=norm_layer, **kwargs)
        in_channels = 4096 if self.backbone.startswith('wideresnet') else 2048
        self.head = ATTENHead(in_channels, nclass, norm_layer, self._up_kwargs, 
                              nheads=nheads, nmixs=nmixs, with_global=with_global,
                              with_enc=with_enc, se_loss=se_loss,
                              lateral=with_lateral)
        if aux:
            self.auxlayer = FCNHead(1024, nclass, norm_layer)

    def forward(self, x):
        imsize = x.size()[2:]
        #_, _, c3, c4 = self.base_forward(x)
        #x = list(self.head(c4))
        features = self.base_forward(x)
        x = list(self.head(*features))
        x[0] = interpolate(x[0], imsize, **self._up_kwargs)
        if self.aux:
            #auxout = self.auxlayer(c3)
            auxout = self.auxlayer(features[2])
            auxout = interpolate(auxout, imsize, **self._up_kwargs)
            x.append(auxout)
        return tuple(x)

    def demo(self, x):
        imsize = x.size()[2:]
        features = self.base_forward(x)
        return self.head.demo(*features)

class GlobalPooling(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, up_kwargs):
        super(GlobalPooling, self).__init__()
        self._up_kwargs = up_kwargs
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                 norm_layer(out_channels),
                                 nn.ReLU(True))

    def forward(self, x):
        _, _, h, w = x.size()
        pool = self.gap(x)
        return interpolate(pool, (h,w), **self._up_kwargs)
 
class ATTENHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, up_kwargs,
                 nheads, nmixs, with_global,
                 with_enc, se_loss, lateral):
        super(ATTENHead, self).__init__()
        self.with_enc = with_enc
        self.se_loss = se_loss
        self._up_kwargs = up_kwargs
        inter_channels = in_channels // 4
        self.lateral = lateral
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels),
            nn.ReLU())
        if lateral:
            self.connect = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(512, 512, kernel_size=1, bias=False),
                    norm_layer(512),
                    nn.ReLU(inplace=True)),
                nn.Sequential(
                    nn.Conv2d(1024, 512, kernel_size=1, bias=False),
                    norm_layer(512),
                    nn.ReLU(inplace=True)),
            ])
            self.fusion = nn.Sequential(
                    nn.Conv2d(3*512, 512, kernel_size=3, padding=1, bias=False),
                    norm_layer(512),
                    nn.ReLU(inplace=True))
        extended_channels = 0
        self.atten = ACFModule(nheads, nmixs, inter_channels, inter_channels//nheads*nmixs,
                               inter_channels//nheads, norm_layer)
        if with_global:
            extended_channels = inter_channels
            self.atten_layers = ConcurrentModule([
                    GlobalPooling(inter_channels, extended_channels, norm_layer, self._up_kwargs),
                    self.atten,
                    #nn.Sequential(*atten),
                ])
        else:
            self.atten_layers = nn.Sequential(*atten)
        if with_enc:
            self.encmodule = EncModule(inter_channels+extended_channels, out_channels, ncodes=32,
                                       se_loss=se_loss, norm_layer=norm_layer)
        self.conv6 = nn.Sequential(nn.Dropout(0.1, False),
                                   nn.Conv2d(inter_channels+extended_channels, out_channels, 1))

    def forward(self, *inputs):
        feat = self.conv5(inputs[-1])
        if self.lateral:
            c2 = self.connect[0](inputs[1])
            c3 = self.connect[1](inputs[2])
            feat = self.fusion(torch.cat([feat, c2, c3], 1))
        feat = self.atten_layers(feat)
        if self.with_enc:
            outs = list(self.encmodule(feat))
        else:
            outs = [feat]
        outs[0] = self.conv6(outs[0])
        return tuple(outs)

    def demo(self, *inputs):
        feat = self.conv5(inputs[-1])
        if self.lateral:
            c2 = self.connect[0](inputs[1])
            c3 = self.connect[1](inputs[2])
            feat = self.fusion(torch.cat([feat, c2, c3], 1))
        attn = self.atten.demo(feat)
        return attn

def get_atten(dataset='pascal_voc', backbone='resnet50s', pretrained=False,
              root='~/.encoding/models', **kwargs):
    r"""ATTEN model from the paper `"Fully Convolutional Network for semantic segmentation"
    <https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_atten.pdf>`_
    Parameters
    ----------
    dataset : str, default pascal_voc
        The dataset that model pretrained on. (pascal_voc, ade20k)
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    pooling_mode : str, default 'avg'
        Using 'max' pool or 'avg' pool in the Attention module.
    root : str, default '~/.encoding/models'
        Location for keeping the model parameters.
    Examples
    --------
    >>> model = get_atten(dataset='pascal_voc', backbone='resnet50s', pretrained=False)
    >>> print(model)
    """
    # infer number of classes
    from ...datasets import datasets, acronyms
    model = ATTEN(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, **kwargs)
    if pretrained:
        from .model_store import get_model_file
        model.load_state_dict(torch.load(
            get_model_file('atten_%s_%s'%(backbone, acronyms[dataset]), root=root)))
    return model
