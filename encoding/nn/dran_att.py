###########################################################################
# Created by: CASIA IVA  
# Email: jliu@nlpr.ia.ac.cn 
# Copyright (c) 2020
###########################################################################
import numpy as np
import torch
import torch.nn as nn
import math
from .syncbn import SyncBatchNorm
from torch.nn.functional import upsample
from torch.nn import Module, Sequential, Conv2d, ReLU,AdaptiveMaxPool2d, AdaptiveAvgPool2d, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout,PairwiseDistance 
from torch.nn import functional as F
from torch.autograd import Variable

up_kwargs = {'mode': 'bilinear', 'align_corners': True}
__all__ = ['CPAMEnc', 'CPAMDec','CCAMDec', 'CLGD']


class CPAMEnc(Module):
    """
    CPAM encoding module
    """
    def __init__(self, in_channels, norm_layer):
        super(CPAMEnc, self).__init__()
        self.pool1 = AdaptiveAvgPool2d(1)
        self.pool2 = AdaptiveAvgPool2d(2)
        self.pool3 = AdaptiveAvgPool2d(3)
        self.pool4 = AdaptiveAvgPool2d(6)

        self.conv1 = Sequential(Conv2d(in_channels, in_channels, 1, bias=False),
                                norm_layer(in_channels),
                                ReLU(True))
        self.conv2 = Sequential(Conv2d(in_channels, in_channels, 1, bias=False),
                                norm_layer(in_channels),
                                ReLU(True))
        self.conv3 = Sequential(Conv2d(in_channels, in_channels, 1, bias=False),
                                norm_layer(in_channels),
                                ReLU(True))
        self.conv4 = Sequential(Conv2d(in_channels, in_channels, 1, bias=False),
                                norm_layer(in_channels),
                                ReLU(True))

    def forward(self, x):
        b, c, h, w = x.size()
        
        feat1 = self.conv1(self.pool1(x)).view(b,c,-1)
        feat2 = self.conv2(self.pool2(x)).view(b,c,-1)
        feat3 = self.conv3(self.pool3(x)).view(b,c,-1)
        feat4 = self.conv4(self.pool4(x)).view(b,c,-1)
        
        return torch.cat((feat1, feat2, feat3, feat4), 2)


class CPAMDec(Module):
    """
    CPAM decoding module
    """
    def __init__(self,in_channels):
        super(CPAMDec,self).__init__()
        self.softmax  = Softmax(dim=-1)
        self.scale = Parameter(torch.zeros(1))

        self.conv_query = Conv2d(in_channels = in_channels , out_channels = in_channels//4, kernel_size= 1) # query_conv2
        self.conv_key = Linear(in_channels, in_channels//4) # key_conv2
        self.conv_value = Linear(in_channels, in_channels) # value2
    def forward(self, x,y):
        """
            inputs :
                x : input feature(N,C,H,W) y:gathering centers(N,K,M)
            returns :
                out : compact position attention feature
                attention map: (H*W)*M
        """
        m_batchsize,C,width ,height = x.size()
        m_batchsize,K,M = y.size()

        proj_query  = self.conv_query(x).view(m_batchsize,-1,width*height).permute(0,2,1)#BxNxd
        proj_key =  self.conv_key(y).view(m_batchsize,K,-1).permute(0,2,1)#BxdxK
        energy =  torch.bmm(proj_query,proj_key)#BxNxK
        attention = self.softmax(energy) #BxNxk

        proj_value = self.conv_value(y).permute(0,2,1) #BxCxK
        out = torch.bmm(proj_value,attention.permute(0,2,1))#BxCxN
        out = out.view(m_batchsize,C,width,height)
        out = self.scale*out + x
        return out


class CCAMDec(Module):
    """
    CCAM decoding module
    """
    def __init__(self):
        super(CCAMDec,self).__init__()
        self.softmax  = Softmax(dim=-1)
        self.scale = Parameter(torch.zeros(1))

    def forward(self, x,y):
        """
            inputs :
                x : input feature(N,C,H,W) y:gathering centers(N,K,H,W)
            returns :
                out : compact channel attention feature
                attention map: K*C
        """
        m_batchsize,C,width ,height = x.size()
        x_reshape =x.view(m_batchsize,C,-1)

        B,K,W,H = y.size()
        y_reshape =y.view(B,K,-1)
        proj_query  = x_reshape #BXC1XN
        proj_key  = y_reshape.permute(0,2,1) #BX(N)XC
        energy =  torch.bmm(proj_query,proj_key) #BXC1XC
        energy_new = torch.max(energy,-1,keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = y.view(B,K,-1) #BCN
        
        out = torch.bmm(attention,proj_value) #BC1N
        out = out.view(m_batchsize,C,width ,height)

        out = x + self.scale*out
        return out


class CLGD(Module):
    """
    Cross-level Gating Decoder
    """
    def __init__(self, in_channels, out_channels, norm_layer):
        super(CLGD, self).__init__()

        inter_channels= 32
        self.conv_low = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                norm_layer(inter_channels),
                                nn.ReLU()) #skipconv
        
        self.conv_cat = nn.Sequential(nn.Conv2d(in_channels+inter_channels, in_channels, 3, padding=1, bias=False),
                                     norm_layer(in_channels),
                                     nn.ReLU()) # fusion1

        self.conv_att = nn.Sequential(nn.Conv2d(in_channels+inter_channels, 1, 1),
                                    nn.Sigmoid()) # att

        self.conv_out = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                                     norm_layer(out_channels),
                                     nn.ReLU()) # fusion2
        self._up_kwargs = up_kwargs

        self.gamma = nn.Parameter(torch.ones(1))

    def forward(self, x,y):
        """
            inputs :
                x : low level feature(N,C,H,W)  y:high level feature(N,C,H,W)
            returns :
                out :  cross-level gating decoder feature
        """
        low_lvl_feat = self.conv_low(x)
        high_lvl_feat = upsample(y, low_lvl_feat.size()[2:], **self._up_kwargs)
        feat_cat = torch.cat([low_lvl_feat,high_lvl_feat],1)
      
        low_lvl_feat_refine = self.gamma*self.conv_att(feat_cat)*low_lvl_feat 
        low_high_feat = torch.cat([low_lvl_feat_refine,high_lvl_feat],1)
        low_high_feat = self.conv_cat(low_high_feat)
        
        low_high_feat = self.conv_out(low_high_feat)

        return low_high_feat