###########################################################################
# Created by: Hang Zhang 
# Email: zhang.hang@rutgers.edu 
# Copyright (c) 2017
###########################################################################

import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.nn.functional import upsample

import encoding
from .base import BaseNet
from .fcn import FCNHead

__all__ = ['EncNet', 'EncModule', 'get_encnet', 'get_encnet_resnet50_pcontext',
           'get_encnet_resnet101_pcontext']

class EncNet(BaseNet):
    def __init__(self, nclass, backbone, aux=True, se_loss=True,
                 norm_layer=nn.BatchNorm2d, **kwargs):
        super(EncNet, self).__init__(nclass, backbone, aux, se_loss, norm_layer=norm_layer)
        self.head = EncHead(self.nclass, in_channels=2048, se_loss=se_loss,
                            norm_layer=norm_layer, up_kwargs=self._up_kwargs)
        if aux:
            self.auxlayer = FCNHead(1024, nclass, norm_layer=norm_layer)

    def forward(self, x):
        imsize = x.size()[2:]
        #features = self.base_forward(x)
        _, _, c3, c4 = self.base_forward(x)

        x = list(self.head(c4))
        x[0] = upsample(x[0], imsize, **self._up_kwargs)
        if self.aux:
            auxout = self.auxlayer(c3)
            auxout = upsample(auxout, imsize, **self._up_kwargs)
            x.append(auxout)
        return tuple(x)


class EncModule(nn.Module):
    def __init__(self, in_channels, nclass, ncodes=32, se_loss=True, norm_layer=None):
        super(EncModule, self).__init__()
        if isinstance(norm_layer, encoding.nn.BatchNorm2d):
            norm_layer = encoding.nn.BatchNorm1d
        else:
            norm_layer = nn.BatchNorm1d
        self.se_loss = se_loss
        self.encoding = nn.Sequential(
            encoding.nn.Encoding(D=in_channels, K=ncodes),
            norm_layer(ncodes),
            nn.ReLU(inplace=True),
            encoding.nn.Sum(dim=1))
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.Sigmoid())
        if self.se_loss:
            self.selayer = nn.Linear(in_channels, nclass)

    def forward(self, x):
        en = self.encoding(x)
        b, c, _, _ = x.size()
        gamma = self.fc(en)
        y = gamma.view(b, c, 1, 1)
        # residual ?
        outputs = [x + x * y]
        if self.se_loss:
            outputs.append(self.selayer(en))
        return tuple(outputs)


class EncHead(nn.Module):
    def __init__(self, out_channels, in_channels, se_loss=True,
                 norm_layer=None, up_kwargs=None):
        super(EncHead, self).__init__()
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels, 512, 3, padding=1, bias=False),
            norm_layer(512),
            nn.ReLU(True))
        self.encmodule = EncModule(512, out_channels, ncodes=32,
            se_loss=se_loss, norm_layer=norm_layer)
        self.dropout = nn.Dropout2d(0.1, False)
        self.conv6 = nn.Conv2d(512, out_channels, 1)
        self.se_loss = se_loss

    def forward(self, x):
        x = self.conv5(x)
        outs = list(self.encmodule(x))
        outs[0] = self.conv6(self.dropout(outs[0]))
        return tuple(outs)


def get_encnet(dataset='pascal_voc', backbone='resnet50', pretrained=False,
               root='~/.encoding/models', **kwargs):
    r"""EncNet model from the paper `"Context Encoding for Semantic Segmentation"
    <https://arxiv.org/pdf/1803.08904.pdf>`_

    Parameters
    ----------
    dataset : str, default pascal_voc
        The dataset that model pretrained on. (pascal_voc, ade20k)
    backbone : str, default resnet50
        The backbone network. (resnet50, 101, 152)
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.encoding/models'
        Location for keeping the model parameters.


    Examples
    --------
    >>> model = get_encnet(dataset='pascal_voc', backbone='resnet50', pretrained=False)
    >>> print(model)
    """
    acronyms = {
        'pascal_voc': 'voc',
        'ade20k': 'ade',
        'pcontext': 'pcontext',
    }
    # infer number of classes
    from ..datasets import datasets, VOCSegmentation, VOCAugSegmentation, ADE20KSegmentation
    model = EncNet(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, **kwargs)
    if pretrained:
        from .model_store import get_model_file
        model.load_state_dict(torch.load(
            get_model_file('encnet_%s_%s'%(backbone, acronyms[dataset]), root=root)))
    return model

def get_encnet_resnet50_pcontext(pretrained=False, root='~/.encoding/models', **kwargs):
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
    >>> model = get_encnet_resnet50_pcontext(pretrained=True)
    >>> print(model)
    """
    return get_encnet('pcontext', 'resnet50', pretrained)

def get_encnet_resnet101_pcontext(pretrained=False, root='~/.encoding/models', **kwargs):
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
    >>> model = get_encnet_resnet101_pcontext(pretrained=True)
    >>> print(model)
    """
    return get_encnet('pcontext', 'resnet101', pretrained)
