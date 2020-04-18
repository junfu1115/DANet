# code adapted from https://github.com/jfzhang95/pytorch-deeplab-xception/
import math
from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F
from ...nn import SyncBatchNorm, GlobalAvgPool2d
from ..model_store import get_model_file

__all__ = ['Xception65', 'Xception71', 'xception65']

def fixed_padding(inputs, kernel_size, dilation):
    kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padded_inputs = F.pad(inputs, (pad_beg, pad_end, pad_beg, pad_end))
    return padded_inputs


class SeparableConv2d(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False, norm_layer=None):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size, stride, 0, dilation,
                               groups=inplanes, bias=bias)
        self.bn = norm_layer(inplanes)
        self.pointwise = nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = fixed_padding(x, self.conv1.kernel_size[0], dilation=self.conv1.dilation[0])
        x = self.conv1(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self, inplanes, planes, reps, stride=1, dilation=1, norm_layer=None,
                 start_with_relu=True, grow_first=True, is_last=False):
        super(Block, self).__init__()
        if planes != inplanes or stride != 1:
            self.skip = nn.Conv2d(inplanes, planes, 1, stride=stride, bias=False)
            self.skipbn = norm_layer(planes)
        else:
            self.skip = None
        self.relu = nn.ReLU(inplace=True)
        rep = []
        filters = inplanes
        if grow_first:
            if start_with_relu:
                rep.append(self.relu)
            rep.append(SeparableConv2d(inplanes, planes, 3, 1, dilation, norm_layer=norm_layer))
            rep.append(norm_layer(planes))
            filters = planes
        for i in range(reps - 1):
            if grow_first or start_with_relu:
                rep.append(self.relu)
            rep.append(SeparableConv2d(filters, filters, 3, 1, dilation, norm_layer=norm_layer))
            rep.append(norm_layer(filters))
        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(inplanes, planes, 3, 1, dilation, norm_layer=norm_layer))
            rep.append(norm_layer(planes))
        if stride != 1:
            rep.append(self.relu)
            rep.append(SeparableConv2d(planes, planes, 3, 2, norm_layer=norm_layer))
            rep.append(norm_layer(planes))
        elif is_last:
            rep.append(self.relu)
            rep.append(SeparableConv2d(planes, planes, 3, 1, dilation, norm_layer=norm_layer))
            rep.append(norm_layer(planes))
        #if not start_with_relu:
        #    rep = rep[1:]
        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)
        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp
        x = x + skip
        return x

class Xception65(nn.Module):
    """Modified Aligned Xception
    """
    def __init__(self, output_stride=32, norm_layer=nn.BatchNorm2d):
        super(Xception65, self).__init__()

        if output_stride == 32:
            entry_block3_stride = 2
            middle_block_dilation = 1
            exit_block20_stride = 2
            exit_block_dilations = (1, 1)
        elif output_stride == 16:
            entry_block3_stride = 2
            middle_block_dilation = 1
            exit_block20_stride = 1
            exit_block_dilations = (1, 2)
        elif output_stride == 8:
            entry_block3_stride = 1
            middle_block_dilation = 2
            exit_block20_stride = 1
            exit_block_dilations = (2, 4)
        else:
            raise NotImplementedError

        # Entry flow
        self.conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False)
        self.bn1 = norm_layer(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False)
        self.bn2 = norm_layer(64)

        self.block1 = Block(64, 128, reps=2, stride=2, norm_layer=norm_layer, start_with_relu=False)
        self.block2 = Block(128, 256, reps=2, stride=2, norm_layer=norm_layer, start_with_relu=False,
                            grow_first=True)
        #print('self.block2', self.block2)
        self.block3 = Block(256, 728, reps=2, stride=entry_block3_stride, norm_layer=norm_layer,
                            start_with_relu=True, grow_first=True, is_last=True)

        # Middle flow
        midflowblocks = []
        for i in range(4, 20):
            midflowblocks.append(('block%d'%i, Block(728, 728, reps=3, stride=1,
                                                     dilation=middle_block_dilation,
                                                     norm_layer=norm_layer, start_with_relu=True,
                                                     grow_first=True)))
        self.midflow = nn.Sequential(OrderedDict(midflowblocks))

        # Exit flow
        self.block20 = Block(728, 1024, reps=2, stride=exit_block20_stride, dilation=exit_block_dilations[0],
                             norm_layer=norm_layer, start_with_relu=True, grow_first=False, is_last=True)

        self.conv3 = SeparableConv2d(1024, 1536, 3, stride=1, dilation=exit_block_dilations[1], norm_layer=norm_layer)
        self.bn3 = norm_layer(1536)

        self.conv4 = SeparableConv2d(1536, 1536, 3, stride=1, dilation=exit_block_dilations[1], norm_layer=norm_layer)
        self.bn4 = norm_layer(1536)

        self.conv5 = SeparableConv2d(1536, 2048, 3, stride=1, dilation=exit_block_dilations[1], norm_layer=norm_layer)
        self.bn5 = norm_layer(2048)

        self.avgpool = GlobalAvgPool2d()
        self.fc = nn.Linear(2048, 1000)

        # Init weights
        self._init_weight()

    def forward(self, x):
        # Entry flow
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.block1(x)
        # add relu here
        x = self.relu(x)
        #c1 = x
        x = self.block2(x)
        #c2 = x
        x = self.block3(x)

        # Middle flow
        x = self.midflow(x)
        #c3 = x

        # Exit flow
        x = self.block20(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)
 
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, SyncBatchNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class Xception71(nn.Module):
    """Modified Aligned Xception
    """
    def __init__(self, output_stride=32, norm_layer=nn.BatchNorm2d):
        super(Xception71, self).__init__()

        if output_stride == 32:
            entry_block3_stride = 2
            middle_block_dilation = 1
            exit_block20_stride = 2
            exit_block_dilations = (1, 1)
        elif output_stride == 16:
            entry_block3_stride = 2
            middle_block_dilation = 1
            exit_block20_stride = 1
            exit_block_dilations = (1, 2)
        elif output_stride == 8:
            entry_block3_stride = 1
            middle_block_dilation = 2
            exit_block20_stride = 1
            exit_block_dilations = (2, 4)
        else:
            raise NotImplementedError

        # Entry flow
        self.conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False)
        self.bn1 = norm_layer(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False)
        self.bn2 = norm_layer(64)

        self.block1 = Block(64, 128, reps=2, stride=2, norm_layer=norm_layer, start_with_relu=False)
        block2 = []
        block2.append(Block(128, 256, reps=2, stride=1, norm_layer=norm_layer, start_with_relu=False,
                            grow_first=True))
        block2.append(Block(256, 256, reps=2, stride=2, norm_layer=norm_layer, start_with_relu=False,
                            grow_first=True))
        block2.append(Block(256, 728, reps=2, stride=1, norm_layer=norm_layer, start_with_relu=False,
                            grow_first=True))
        self.block2 = nn.Sequential(*block2)
        self.block3 = Block(728, 728, reps=2, stride=entry_block3_stride, norm_layer=norm_layer,
                            start_with_relu=True, grow_first=True, is_last=True)

        # Middle flow
        midflowblocks = []
        for i in range(4, 20):
            midflowblocks.append(('block%d'%i, Block(728, 728, reps=3, stride=1,
                                                     dilation=middle_block_dilation,
                                                     norm_layer=norm_layer, start_with_relu=True,
                                                     grow_first=True)))
        self.midflow = nn.Sequential(OrderedDict(midflowblocks))

        # Exit flow
        self.block20 = Block(728, 1024, reps=2, stride=exit_block20_stride, dilation=exit_block_dilations[0],
                             norm_layer=norm_layer, start_with_relu=True, grow_first=False, is_last=True)

        self.conv3 = SeparableConv2d(1024, 1536, 3, stride=1, dilation=exit_block_dilations[1], norm_layer=norm_layer)
        self.bn3 = norm_layer(1536)

        self.conv4 = SeparableConv2d(1536, 1536, 3, stride=1, dilation=exit_block_dilations[1], norm_layer=norm_layer)
        self.bn4 = norm_layer(1536)

        self.conv5 = SeparableConv2d(1536, 2048, 3, stride=1, dilation=exit_block_dilations[1], norm_layer=norm_layer)
        self.bn5 = norm_layer(2048)

        self.avgpool = GlobalAvgPool2d()
        self.fc = nn.Linear(2048, 1000)

        # Init weights
        self._init_weight()

    def forward(self, x):
        # Entry flow
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.block1(x)
        # add relu here
        x = self.relu(x)
        low_level_feat = x
        x = self.block2(x)
        x = self.block3(x)

        # Middle flow
        x = self.midflow(x)

        # Exit flow
        x = self.block20(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x#, low_level_feat

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, SyncBatchNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def xception65(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Xception65(**kwargs)
    if pretrained:
        model.load_state_dict(torch.load(get_model_file('xception65', root=root)))
    return model
