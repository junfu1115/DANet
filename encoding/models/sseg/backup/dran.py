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
from ...nn import PyramidCode,Self_Decoder2,Self_CB_RedRes3,Self_Decoder_Cha_Max
import encoding
from .base import BaseNet
from ipdb import set_trace as st

up_kwargs = {'mode': 'bilinear', 'align_corners': True}
__all__ = ['Dran', 'get_dran']

class Dran(BaseNet):
    r"""Fully Convolutional Networks for Semantic Segmentation

    Paca4u52rameters
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
    >>> model = FCN(nclass=21, backbone='resnet50')
    >>> print(model)
    """
    def __init__(self, nclass, backbone, cut_loss=True,aux=True, se_loss=False, norm_layer=nn.BatchNorm2d, **kwargs):
        super(Dran, self).__init__(nclass, backbone, aux, se_loss, norm_layer=norm_layer, **kwargs)
        self.head = FCNHeadPaca4u52(2048, nclass, norm_layer)
        if aux:
            self.auxlayer = FCNHeadPaca4u52Aux(1024, nclass, norm_layer)
        self.cut_loss = cut_loss

    def forward(self, x):
        imsize = x.size()[2:]
        multix = self.base_forward(x)
        x = list(self.head(multix))
        x[0] = upsample(x[0], imsize, **self._up_kwargs)
        x[1] = upsample(x[1], imsize, **self._up_kwargs)
        outputs = [x[1]]
        outputs.append(x[0])

        #if self.aux:
        #    auxout = self.auxlayer(c3)
        #    auxout = upsample(auxout, imsize, **self._up_kwargs)
        #    outputs.append(auxout)
        return tuple(outputs)


    
class FCNHeadPaca4u52(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer):
        super(FCNHeadPaca4u52, self).__init__()
        inter_channels = in_channels // 4
        self.conv5_s = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU())
        self.en_s = PyramidCode(512,norm_layer)
        #self.cb = Self_CB_RedRes3(50)
        self.de_s = Self_Decoder2(512)

        self.conv5_c = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU())
        self.conv51_c = nn.Sequential(nn.Conv2d(inter_channels, inter_channels//16, 1, bias=False),
                                   norm_layer(inter_channels//16),
                                   nn.ReLU())
        self.de_c = Self_Decoder_Cha_Max()

    
        self.conv51 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU())
        self.conv52 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU())
        self.conv_f = nn.Sequential(nn.Conv2d(inter_channels*2, inter_channels//2, 3, padding=1, bias=False),
                                   norm_layer(inter_channels//2),
                                   nn.ReLU())
 
        self.skipconv = nn.Sequential(nn.Conv2d(256, 32, 3, padding=1, bias=False),
                                        norm_layer(32),
                                        nn.ReLU())
        
        self.fusion = nn.Sequential(nn.Conv2d(288, 256, 3, padding=1, bias=False),
                                     norm_layer(256),
                                     nn.ReLU())

        self.fusion2 = nn.Sequential(nn.Conv2d(256, 256, 3, padding=1, bias=False),
                                     norm_layer(256),
                                     nn.ReLU())


       

        self.att = nn.Sequential(nn.Conv2d(288, 1, 1),
                                    nn.Sigmoid())
                         
        self._up_kwargs = up_kwargs
        self.conv6 = nn.Sequential(nn.Dropout2d(0.1, False),
                                   nn.Conv2d(256, out_channels, 1))
        self.gamma = nn.Parameter(torch.ones(1))
        self.conv7 = nn.Sequential(nn.Conv2d(1024, 256, 3, padding=1, bias=False),
                                   norm_layer(256),
                                   nn.ReLU(),
                                   nn.Dropout2d(0.1, False),
                                   nn.Conv2d(256, out_channels, 1))

    def forward(self, multix):

        feat_c = self.conv5_c(multix[-1])
        feat_en_c = self.conv51_c(feat_c)

        #round1
        #feat_en1 = self.en(feat,feat_en)
        feat_sa_c = self.de_c(feat_c,feat_en_c)        
        
        
        feat_s = self.conv5_s(multix[-1])
        feat_en_s = self.en_s(feat_s).permute(0,2,1)#BKD
        feat_sa_s = self.de_s(feat_s,feat_en_s)

        sc_conv = self.conv51(feat_sa_c)
        ss_conv = self.conv52(feat_sa_s)
        feat_sum = self.conv_f(torch.cat([ss_conv,sc_conv],1))
        

        #UP
        x2 = self.skipconv(multix[0])
        feat_up = upsample(feat_sum, x2.size()[2:], **self._up_kwargs)
        feat_up_cat = torch.cat([x2,feat_up],1)
      
        x2_refine = self.gamma*self.att(feat_up_cat)*x2 
        feat_up_cat_re = torch.cat([x2_refine,feat_up],1)
        feat_up_fu = self.fusion(feat_up_cat_re)
        
        feat_up_fu2 = self.fusion2(feat_up_fu)        
        
        out = self.conv7(multix[-2])
        out2 = self.conv6(feat_up_fu2)
        output = [out]
        output.append(out2)
        return tuple(output)

class FCNHeadPaca4u52Aux(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer):
        super(FCNHeadPaca4u52Aux, self).__init__()
        inter_channels = in_channels // 4
        self.conv5 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU(),
                                   nn.Dropout2d(0.1, False),
                                   nn.Conv2d(inter_channels, out_channels, 1))

    def forward(self, x):
        return self.conv5(x)

def get_dran(dataset='pascal_voc', backbone='resnet50', pretrained=False,
            root='~/.encoding/models', **kwargs):

    r"""FCN model from the paper `"Fully Convolutional Network for semantic segmentation"
    <https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf>`_
    Paca4u52rameters
    ----------
    dataset : str, default pascal_voc
        The dataset that model pretrained on. (pascal_voc, ade20k)
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.encoding/models'
        Location for keeping the model parameters.
    Examples
    --------
    >>> model = get_fcn(dataset='pascal_voc', backbone='resnet50', pretrained=False)
    >>> print(model)
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