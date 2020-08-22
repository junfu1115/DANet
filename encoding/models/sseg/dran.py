###########################################################################
# Created by: CASIA IVA  
# Email: jliu@nlpr.ia.ac.cn 
# Copyright (c) 2020
###########################################################################
from __future__ import division
import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import upsample
from ...nn import CPAMDec,CCAMDec,CPAMEnc, CLGD
import encoding
from .base import BaseNet
from ipdb import set_trace as st

up_kwargs = {'mode': 'bilinear', 'align_corners': True}
__all__ = ['Dran', 'get_dran']

class Dran(BaseNet):
    r"""
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
    >>> model = Dran(nclass=21, backbone='resnet50')
    >>> print(model)
    """
    def __init__(self, nclass, backbone, cut_loss=True,aux=True, se_loss=False, norm_layer=nn.BatchNorm2d, **kwargs):
        super(Dran, self).__init__(nclass, backbone, aux, se_loss, norm_layer=norm_layer, **kwargs)
        self.head = DranHead(2048, nclass, norm_layer)
        in_channels = 256
        if aux:
            # self.auxlayer = DranHeadAux(1024, nclass, norm_layer)
            self.cls_aux = nn.Sequential(nn.Conv2d(1024, in_channels, 3, padding=1, bias=False),
                           norm_layer(in_channels),
                           nn.ReLU(),
                           nn.Dropout2d(0.1, False),
                           nn.Conv2d(in_channels, nclass, 1))

        self.cls_seg = nn.Sequential(nn.Dropout2d(0.1, False),
                   nn.Conv2d(in_channels, nclass, 1))

    def forward(self, x):
        imsize = x.size()[2:]
        multix = self.base_forward(x)

        ## dran head for seg
        final_feat = self.head(multix)
        cls_seg = self.cls_seg(final_feat)
        cls_seg = upsample(cls_seg, imsize, **self._up_kwargs)

        ## aux head for seg
        outputs = [cls_seg]
        if self.aux:
            cls_aux = self.cls_aux(multix[-2])
            cls_aux = upsample(cls_aux, imsize, **self._up_kwargs)
            outputs.append(cls_aux)

        return tuple(outputs)


    
class DranHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer):
        super(DranHead, self).__init__()
        inter_channels = in_channels // 4

        ## Convs or modules for CPAM 
        self.conv_cpam_b = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU()) # conv5_s
        self.cpam_enc = CPAMEnc(inter_channels, norm_layer) # en_s
        self.cpam_dec = CPAMDec(inter_channels) # de_s
        self.conv_cpam_e = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                           norm_layer(inter_channels),
                           nn.ReLU()) # conv52

        ## Convs or modules for CCAM
        self.conv_ccam_b = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU()) # conv5_c
        self.ccam_enc = nn.Sequential(nn.Conv2d(inter_channels, inter_channels//16, 1, bias=False),
                                   norm_layer(inter_channels//16),
                                   nn.ReLU()) # conv51_c
        self.ccam_dec = CCAMDec() # de_c
        self.conv_ccam_e = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU()) # conv51

        ## Fusion conv
        self.conv_cat = nn.Sequential(nn.Conv2d(inter_channels*2, inter_channels//2, 3, padding=1, bias=False),
                                   norm_layer(inter_channels//2),
                                   nn.ReLU()) # conv_f
        ## Cross-level Gating Decoder(CLGD) 
        self.clgd = CLGD(inter_channels//2,inter_channels//2,norm_layer)

    def forward(self, multix):

        ## Compact Channel Attention Module(CCAM)
        ccam_b = self.conv_ccam_b(multix[-1])
        ccam_f = self.ccam_enc(ccam_b)
        ccam_feat = self.ccam_dec(ccam_b,ccam_f)        
        
        ## Compact Spatial Attention Module(CPAM)
        cpam_b = self.conv_cpam_b(multix[-1])
        cpam_f = self.cpam_enc(cpam_b).permute(0,2,1)#BKD
        cpam_feat = self.cpam_dec(cpam_b,cpam_f)

        ## Fuse two modules
        ccam_feat = self.conv_ccam_e(ccam_feat)
        cpam_feat = self.conv_cpam_e(cpam_feat)
        feat_sum = self.conv_cat(torch.cat([cpam_feat,ccam_feat],1))
        
        ## Cross-level Gating Decoder(CLGD) 
        final_feat = self.clgd(multix[0], feat_sum)

        return final_feat


def get_dran(dataset='pascal_voc', backbone='resnet50', pretrained=False,
            root='~/.encoding/models', **kwargs):

    r"""Scene Segmentation with Dual Relation-aware Attention Network
    """
    acronyms = {
        'pascal_voc': 'voc',
        'pascal_aug': 'voc',
        'pcontext': 'pcontext',
        'ade20k': 'ade',
    }
    # infer number of classes
    from ...datasets import datasets, VOCSegmentation, VOCAugSegmentation, ADE20KSegmentation
    model = Dran(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root, **kwargs)
    if pretrained:
        from .model_store import get_model_file
        model.load_state_dict(torch.load(
            get_model_file('fcn_%s_%s'%(backbone, acronyms[dataset]), root=root)),
            strict= False)
    return model
