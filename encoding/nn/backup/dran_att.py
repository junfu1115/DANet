###########################################################################
# Created by: Hang Zhang 
# Email: zhang.hang@rutgers.edu 
# Copyright (c) 2018
###########################################################################
import numpy as np
import torch
import math
from .syncbn import SyncBatchNorm
from torch.nn import Module, Sequential, Conv2d, ReLU,AdaptiveMaxPool2d, AdaptiveAvgPool2d, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout,PairwiseDistance 
from torch.nn import functional as F
from torch.autograd import Variable


__all__ = ['PyramidCode', 'Self_Decoder2','Self_CB_RedRes3','Self_Decoder_Cha_Max']


class PyramidCode(Module):
    """
    Reference:
        Zhao, Hengshuang, et al. *"Pyramid scene parsing network."*
    """
    def __init__(self, in_channels, norm_layer):
        super(PyramidCode, self).__init__()
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
        #feat1 = F.upsample(self.conv1(self.pool1(x)), (h, w), **self._up_kwargs)
        #feat2 = F.upsample(self.conv2(self.pool2(x)), (h, w), **self._up_kwargs)
        #feat3 = F.upsample(self.conv3(self.pool3(x)), (h, w), **self._up_kwargs)
        #feat4 = F.upsample(self.conv4(self.pool4(x)), (h, w), **self._up_kwargs)
        #return torch.cat((x, feat1, feat2, feat3, feat4), 1)
        
        feat1 = self.conv1(self.pool1(x)).view(b,c,-1)
        feat2 = self.conv2(self.pool2(x)).view(b,c,-1)
        feat3 = self.conv3(self.pool3(x)).view(b,c,-1)
        feat4 = self.conv4(self.pool4(x)).view(b,c,-1)
        
        return torch.cat((feat1, feat2, feat3, feat4), 2)


class Self_Decoder2(Module):
    def __init__(self,D):
        super(Self_Decoder2,self).__init__()
        self.softmax  = Softmax(dim=-1)
        self.scale = Parameter(torch.zeros(1))

        self.query_conv2 = Conv2d(in_channels = D , out_channels = D//4, kernel_size= 1)
        self.key_conv2 = Linear(D, D//4)
        self.value2=Linear(D, D)

    def forward(self, X,y1):
        # input X is a 4D tensor
        m_batchsize,C,width ,height = X.size()
        m_batchsize,K,D = y1.size()

        ##stage2
        proj_query2  = self.query_conv2(X).view(m_batchsize,-1,width*height).permute(0,2,1)#BxNxd
        proj_key2 =  self.key_conv2(y1).view(m_batchsize,K,-1).permute(0,2,1)#BxdxK


        energy2 =  torch.bmm(proj_query2,proj_key2)#BxNxK
        attention2 = self.softmax(energy2) #BxNxk

        #proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        proj_value2 = self.value2(y1).permute(0,2,1) #BxCxK
        out2 = torch.bmm(proj_value2,attention2.permute(0,2,1))#BxCxN
        out2 = out2.view(m_batchsize,C,width,height)
        out2 = self.scale*out2 + X
        return out2

class Self_CB_RedRes3(Module):
    def __init__(self,K):
        super(Self_CB_RedRes3,self).__init__()

        self.dim = Sequential(Linear(K, K//2),
                            ReLU(inplace=True))
        self.linear = Linear(K//2, K)

    def forward(self,y1):
        y11 = self.dim(y1.permute(0,2,1))#BDk
        y21 = self.linear(y11).permute(0,2,1)+y1#BkD
        y11 = y11.permute(0,2,1) 

        return y21,y11



class Self_Decoder_Cha_Max(Module):
    def __init__(self):
        super(Self_Decoder_Cha_Max,self).__init__()
        self.softmax  = Softmax(dim=-1)
        self.scale = Parameter(torch.zeros(1))

    def forward(self, X,y):
        # input X is a 4D tensor
        m_batchsize,C1,width ,height = X.size()
        X1 =X.view(m_batchsize,C1,-1)

        B,C,W,H = y.size()
        y1 =y.view(B,C,-1)



        proj_query  = X1 #BXC1XN


        proj_key  = y1.permute(0,2,1) #BX(N)XC


        energy =  torch.bmm(proj_query,proj_key) #BXC1XC
        energy_new = torch.max(energy,-1,keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = y.view(B,C,-1) #BCN
        

        out = torch.bmm(attention,proj_value) #BC1N
        out = out.view(m_batchsize,C1,width ,height)

        X_out = X + self.scale*out
        return X_out