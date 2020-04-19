##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## ECE Department, Rutgers University
## Email: zhang.hang@rutgers.edu
## Copyright (c) 2017
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import torch
import torch.nn as nn

from ..nn import Encoding, View, Normalize
from .backbone import resnet50s, resnet101s, resnet152s

__all__ = ['DeepTen', 'get_deepten', 'get_deepten_resnet50_minc']

class DeepTen(nn.Module):
    def __init__(self, nclass, backbone):
        super(DeepTen, self).__init__()
        self.backbone = backbone
        # copying modules from pretrained models
        if self.backbone == 'resnet50':
            self.pretrained = resnet50s(pretrained=True, dilated=False)
        elif self.backbone == 'resnet101':
            self.pretrained = resnet101s(pretrained=True, dilated=False)
        elif self.backbone == 'resnet152':
            self.pretrained = resnet152s(pretrained=True, dilated=False)
        else:
            raise RuntimeError('unknown backbone: {}'.format(self.backbone))
        n_codes = 32
        self.head = nn.Sequential(
            nn.Conv2d(2048, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            Encoding(D=128,K=n_codes),
            View(-1, 128*n_codes),
            Normalize(),
            nn.Linear(128*n_codes, nclass),
        )

    def forward(self, x):
        _, _, h, w = x.size()
        x = self.pretrained.conv1(x)
        x = self.pretrained.bn1(x)
        x = self.pretrained.relu(x)
        x = self.pretrained.maxpool(x)
        x = self.pretrained.layer1(x)
        x = self.pretrained.layer2(x)
        x = self.pretrained.layer3(x)
        x = self.pretrained.layer4(x)
        return self.head(x)

def get_deepten(dataset='pascal_voc', backbone='resnet50', pretrained=False,
                root='~/.encoding/models', **kwargs):
    r"""DeepTen model from the paper `"Deep TEN: Texture Encoding Network"
    <https://arxiv.org/pdf/1612.02844v1.pdf>`_
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
    >>> model = get_deepten(dataset='minc', backbone='resnet50', pretrained=False)
    >>> print(model)
    """
    from ..datasets import datasets, acronyms
    model = DeepTen(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, **kwargs)
    if pretrained:
        from .model_store import get_model_file
        model.load_state_dict(torch.load(
            get_model_file('deepten_%s_%s'%(backbone, acronyms[dataset]), root=root)))
    return model

def get_deepten_resnet50_minc(pretrained=False, root='~/.encoding/models', **kwargs):
    r"""DeepTen model from the paper `"Deep TEN: Texture Encoding Network"
    <https://arxiv.org/pdf/1612.02844v1.pdf>`_
    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.encoding/models'
        Location for keeping the model parameters.


    Examples
    --------
    >>> model = get_deepten_resnet50_minc(pretrained=True)
    >>> print(model)
    """
    return get_deepten(dataset='minc', backbone='resnet50', pretrained=pretrained,
                       root=root, **kwargs)
