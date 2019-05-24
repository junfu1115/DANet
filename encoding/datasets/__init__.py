import warnings
from torchvision.datasets import *
from .base import *
from .coco import COCOSegmentation
from .ade20k import ADE20KSegmentation
from .pascal_voc import VOCSegmentation
from .pascal_aug import VOCAugSegmentation
from .pcontext import ContextSegmentation
from .cityscapes import CitySegmentation
from .imagenet import ImageNetDataset
from .minc import MINCDataset

from ..utils import EncodingDeprecationWarning

datasets = {
    'coco': COCOSegmentation,
    'ade20k': ADE20KSegmentation,
    'pascal_voc': VOCSegmentation,
    'pascal_aug': VOCAugSegmentation,
    'pcontext': ContextSegmentation,
    'citys': CitySegmentation,
    'imagenet': ImageNetDataset,
    'minc': MINCDataset,
    'cifar10': CIFAR10,
}

acronyms = {
    'coco': 'coco',
    'pascal_voc': 'voc',
    'pascal_aug': 'voc',
    'pcontext': 'pcontext',
    'ade20k': 'ade',
    'citys': 'citys',
    'minc': 'minc',
    'cifar10': 'cifar10',
}

def get_dataset(name, **kwargs):
    return datasets[name.lower()](**kwargs)

def _make_deprecate(meth, old_name):
    new_name = meth.__name__

    def deprecated_init(*args, **kwargs):
        warnings.warn("encoding.dataset.{} is now deprecated in favor of encoding.dataset.{}."
                      .format(old_name, new_name), EncodingDeprecationWarning)
        return meth(*args, **kwargs)

    deprecated_init.__doc__ = r"""
    {old_name}(...)
    .. warning::
        This method is now deprecated in favor of :func:`torch.nn.init.{new_name}`.
    See :func:`~torch.nn.init.{new_name}` for details.""".format(
        old_name=old_name, new_name=new_name)
    deprecated_init.__name__ = old_name
    return deprecated_init

get_segmentation_dataset = _make_deprecate(get_dataset, 'get_segmentation_dataset')
