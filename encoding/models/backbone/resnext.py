"""ResNeXt models"""

from .resnet import ResNet, Bottleneck
from ..model_store import get_model_file

__all__ = ['resnext50_32x4d', 'resnext101_32x8d']

def resnext50_32x4d(pretrained=False, root='~/.encoding/models', **kwargs):
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['bottleneck_width'] = 4
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(
            get_model_file('resnext50_32x4d', root=root)), strict=False)
    return model

def resnext101_32x8d(pretrained=False, root='~/.encoding/models', **kwargs):
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['bottleneck_width'] = 8
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(
            get_model_file('resnext101_32x8d', root=root)), strict=False)
    return model

