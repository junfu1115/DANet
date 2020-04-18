import sys
from collections import OrderedDict
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...nn import SyncBatchNorm, GlobalAvgPool2d
from ..model_store import get_model_file

__all__ = ['WideResNet', 'wideresnet38', 'wideresnet50']

ABN = partial(SyncBatchNorm, activation='leaky_relu', slope=0.01, sync=True, inplace=True)

class BasicBlock(nn.Module):
    """WideResNet BasicBlock
    """
    def __init__(self, inplanes, planes, stride=1, dilation=1, expansion=1, downsample=None,
                 previous_dilation=1, dropout=0.0, **kwargs):
        super(BasicBlock, self).__init__()
        self.bn1 = ABN(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn2 = ABN(planes)
        self.conv2 = nn.Conv2d(planes, planes * expansion, kernel_size=3,
                               stride=1, padding=previous_dilation, dilation=previous_dilation,
                               bias=False)
        self.downsample = downsample
        self.drop = None
        if dropout > 0.0:
            self.drop = nn.Dropout(dropout)

    def forward(self, x):
        if self.downsample:
            bn1 = self.bn1(x)
            residual = self.downsample(bn1)
        else:
            residual = x.clone()
            bn1 = self.bn1(x)

        out = self.conv1(bn1)
        out = self.bn2(out)
        if self.drop:
            out = self.drops(out)
        out = self.conv2(out)
        out = out + residual
        return out


class Bottleneck(nn.Module):
    """WideResNet BottleneckV1b
    """
    # pylint: disable=unused-argument
    def __init__(self, inplanes, planes, stride=1, dilation=1, expansion=4, dropout=0.0,
                 downsample=None, previous_dilation=1, **kwargs):
        super(Bottleneck, self).__init__()
        self.bn1 = ABN(inplanes)

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn2 = ABN(planes)
        self.conv2 = nn.Conv2d(planes, planes*expansion//2, kernel_size=3, stride=stride,
            padding=dilation, dilation=dilation, bias=False)

        self.bn3 = ABN(planes*expansion//2)
        self.conv3 = nn.Conv2d(planes*expansion//2, planes*expansion, kernel_size=1,
            bias=False)
        self.downsample = downsample
        self.drop = None
        if dropout > 0.0:
            self.drop = nn.Dropout(dropout)

    def forward(self, x):
        if self.downsample:
            bn1 = self.bn1(x)
            residual = self.downsample(bn1)
        else:
            residual = x.clone()
            bn1 = self.bn1(x)

        out = self.conv1(bn1)
        out = self.bn2(out)
        out = self.conv2(out)
        out = self.bn3(out)

        if self.drop:
            out = self.drop(out)

        out = self.conv3(out)
        out = out + residual
        return out


class WideResNet(nn.Module):
    """ Pre-trained WideResNet Model
    featuremaps at conv5.

    Parameters
    ----------
    layers : list of int
        Numbers of layers in each block
    classes : int, default 1000
        Number of classification classes.
    dilated : bool, default False
        Applying dilation strategy to pretrained ResNet yielding a stride-8 model,
        typically used in Semantic Segmentation.
    final_drop : float, default 0.0
        Dropout ratio before the final classification layer.

    Reference:

        - Zifeng Wu, et al. "Wider or Deeper: Revisiting the ResNet Model for Visual Recognition"

        - Samuel Rota Bulò, et al. 
            "In-Place Activated BatchNorm for Memory-Optimized Training of DNNs"
    """

    # pylint: disable=unused-variable
    def __init__(self, layers, classes=1000, dilated=False, **kwargs):
        self.inplanes = 64
        super(WideResNet, self).__init__()
        self.mod1 = nn.Conv2d(3, 64, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.pool2 = nn.MaxPool2d(3, stride=2, padding=1)

        self.mod2 = self._make_layer(2, BasicBlock, 128, layers[0])
        self.pool3 = nn.MaxPool2d(3, stride=2, padding=1)

        self.mod3 = self._make_layer(3, BasicBlock, 256, layers[1], stride=1)
        self.mod4 = self._make_layer(4, BasicBlock, 512, layers[2], stride=2)

        if dilated:
            self.mod5 = self._make_layer(5, BasicBlock, 512, layers[3], stride=1, dilation=2,
                                         expansion=2)
            self.mod6 = self._make_layer(6, Bottleneck, 512, layers[4], stride=1, dilation=4,
                                         expansion=4, dropout=0.3)
            self.mod7 = self._make_layer(7, Bottleneck, 1024, layers[5], stride=1, dilation=4,
                                         expansion=4, dropout=0.5)
        else:
            self.mod5 = self._make_layer(5, BasicBlock, 512, layers[3], stride=2, expansion=2)
            self.mod6 = self._make_layer(6, Bottleneck, 512, layers[4], stride=2,
                                         expansion=4, dropout=0.3)
            self.mod7 = self._make_layer(7, Bottleneck, 1024, layers[5], stride=1, expansion=4,
                                         dropout=0.5)
        self.bn_out = ABN(4096)

        self.avgpool = GlobalAvgPool2d()
        self.fc = nn.Linear(4096, classes)

    def _make_layer(self, stage_index, block, planes, blocks, stride=1, dilation=1, expansion=1,
                    dropout=0.0):
        downsample = None
        if stride != 1 or self.inplanes != planes * expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * expansion,
                          kernel_size=1, stride=stride, bias=False),
            )

        layers = []
        if dilation in (1, 2):
            layers.append(block(self.inplanes, planes, stride, dilation=1, expansion=expansion,
                             dropout=dropout, downsample=downsample, previous_dilation=dilation))
        elif dilation == 4 and stage_index < 7:
            layers.append(block(self.inplanes, planes, stride, dilation=2, expansion=expansion,
                             dropout=dropout, downsample=downsample, previous_dilation=dilation))
        else:
            assert stage_index == 7
            layers.append(block(self.inplanes, planes, stride, dilation=dilation, expansion=expansion,
                             dropout=dropout, downsample=downsample, previous_dilation=dilation))

        self.inplanes = planes * expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, expansion=expansion,
                             dropout=dropout, previous_dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.mod1(x)
        x = self.pool2(x)
        x = self.mod2(x)

        x = self.pool3(x)
        x = self.mod3(x)
        x = self.mod4(x)
        x = self.mod5(x)
        x = self.mod6(x)
        x = self.mod7(x)

        x = self.bn_out(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def wideresnet38(pretrained=False, root='~/.encoding/models', **kwargs):
    """Constructs a WideResNet-38 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = WideResNet([3, 3, 6, 3, 1, 1], **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(
            get_model_file('wideresnet38', root=root)), strict=False)
    return model


def wideresnet50(pretrained=False, root='~/.encoding/models', **kwargs):
    """Constructs a WideResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = WideResNet([3, 3, 6, 6, 3, 1], **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(
            get_model_file('wideresnet50', root=root)), strict=False)
    return model
