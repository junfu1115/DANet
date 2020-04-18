###########################################################################
# Created by: Hang Zhang 
# Email: zhang.hang@rutgers.edu 
# Copyright (c) 2018
###########################################################################
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from .syncbn import SyncBatchNorm

__all__ = ['ACFModule', 'MixtureOfSoftMaxACF']

class ACFModule(nn.Module):
    """ Multi-Head Attention module """
    def __init__(self, n_head, n_mix, d_model, d_k, d_v, norm_layer=SyncBatchNorm, 
                 kq_transform='conv', value_transform='conv',
                 pooling=True, concat=False, dropout=0.1):
        super(ACFModule, self).__init__()

        self.n_head = n_head
        self.n_mix = n_mix
        self.d_k = d_k
        self.d_v = d_v
        self.pooling = pooling
        self.concat = concat

        if self.pooling:
            self.pool = nn.AvgPool2d(3, 2, 1, count_include_pad=False)

        if kq_transform == 'conv':
            self.conv_qs = nn.Conv2d(d_model, n_head*d_k, 1)
            nn.init.normal_(self.conv_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        elif kq_transform == 'ffn':
            self.conv_qs = nn.Sequential(
                nn.Conv2d(d_model, n_head*d_k, 3, padding=1, bias=False),
                norm_layer(n_head*d_k),
                nn.ReLU(True),
                nn.Conv2d(n_head*d_k, n_head*d_k, 1),
            )
            nn.init.normal_(self.conv_qs[-1].weight, mean=0, std=np.sqrt(1.0 / d_k))
        elif kq_transform == 'dffn':
            self.conv_qs = nn.Sequential(
                nn.Conv2d(d_model, n_head*d_k, 3, padding=4, dilation=4, bias=False),
                norm_layer(n_head*d_k),
                nn.ReLU(True),
                nn.Conv2d(n_head*d_k, n_head*d_k, 1),
            )
            nn.init.normal_(self.conv_qs[-1].weight, mean=0, std=np.sqrt(1.0 / d_k))
        else:
            raise NotImplemented
        #self.conv_ks = nn.Conv2d(d_model, n_head*d_k, 1)
        self.conv_ks = self.conv_qs
        if value_transform == 'conv':
            self.conv_vs = nn.Conv2d(d_model, n_head*d_v, 1)
        else:
            raise NotImplemented

        #nn.init.normal_(self.conv_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.conv_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = MixtureOfSoftMaxACF(n_mix=n_mix, d_k=d_k)

        self.conv = nn.Conv2d(n_head*d_v, d_model, 1, bias=False)
        self.norm_layer = norm_layer(d_model)

    def forward(self, x):
        residual = x

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        b_, c_, h_, w_ = x.size()

        if self.pooling:
            qt = self.conv_ks(x).view(b_*n_head, d_k, h_*w_)
            kt = self.conv_ks(self.pool(x)).view(b_*n_head, d_k, h_*w_//4)
            vt = self.conv_vs(self.pool(x)).view(b_*n_head, d_v, h_*w_//4)
        else:
            kt = self.conv_ks(x).view(b_*n_head, d_k, h_*w_)
            qt = kt
            vt = self.conv_vs(x).view(b_*n_head, d_v, h_*w_)

        output, attn = self.attention(qt, kt, vt)

        output = output.transpose(1, 2).contiguous().view(b_, n_head*d_v, h_, w_)

        output = self.conv(output)
        if self.concat:
            output = torch.cat((self.norm_layer(output), residual), 1)
        else:
            output = self.norm_layer(output) + residual
        return output

    def demo(self, x):
        residual = x

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        b_, c_, h_, w_ = x.size()

        if self.pooling:
            qt = self.conv_ks(x).view(b_*n_head, d_k, h_*w_)
            kt = self.conv_ks(self.pool(x)).view(b_*n_head, d_k, h_*w_//4)
            vt = self.conv_vs(self.pool(x)).view(b_*n_head, d_v, h_*w_//4)
        else:
            kt = self.conv_ks(x).view(b_*n_head, d_k, h_*w_)
            qt = kt
            vt = self.conv_vs(x).view(b_*n_head, d_v, h_*w_)

        _, attn = self.attention(qt, kt, vt)
        attn.view(b_, n_head, h_*w_, -1)
        return attn

    def extra_repr(self):
        return 'n_head={}, n_mix={}, d_k={}, pooling={}' \
            .format(self.n_head, self.n_mix, self.d_k, self.pooling)


class MixtureOfSoftMaxACF(nn.Module):
    """"Mixture of SoftMax"""
    def __init__(self, n_mix, d_k, attn_dropout=0.1):
        super(MixtureOfSoftMaxACF, self).__init__()
        self.temperature = np.power(d_k, 0.5)
        self.n_mix = n_mix
        self.att_drop = attn_dropout
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax1 = nn.Softmax(dim=1)
        self.softmax2 = nn.Softmax(dim=2)
        self.d_k = d_k
        if n_mix > 1:
            self.weight = nn.Parameter(torch.Tensor(n_mix, d_k))
            std = np.power(n_mix, -0.5)
            self.weight.data.uniform_(-std, std)

    def forward(self, qt, kt, vt):
        B, d_k, N = qt.size()
        m = self.n_mix
        assert d_k == self.d_k
        d = d_k // m
        if m > 1:
            # \bar{v} \in R^{B, d_k, 1}
            bar_qt = torch.mean(qt, 2, True)
            # pi \in R^{B, m, 1}
            pi = self.softmax1(torch.matmul(self.weight, bar_qt)).view(B*m, 1, 1)
        # reshape for n_mix
        q = qt.view(B*m, d, N).transpose(1, 2)
        N2 = kt.size(2)
        kt = kt.view(B*m, d, N2)
        v = vt.transpose(1, 2)
        # {Bm, N, N}
        attn = torch.bmm(q, kt)
        attn = attn / self.temperature
        attn = self.softmax2(attn)
        attn = self.dropout(attn)
        if m > 1:
            # attn \in R^{Bm, N, N2} => R^{B, N, N2}
            attn = (attn * pi).view(B, m, N, N2).sum(1)
        output = torch.bmm(attn, v)
        return output, attn
