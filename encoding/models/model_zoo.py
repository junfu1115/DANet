# pylint: disable=wildcard-import, unused-wildcard-import

from .backbone import *
from .sseg import *
from .deepten import *

__all__ = ['get_model']

def get_model(name, **kwargs):
    """Returns a pre-defined model by name

    Parameters
    ----------
    name : str
        Name of the model.
    pretrained : bool
        Whether to load the pretrained weights for model.
    root : str, default '~/.encoding/models'
        Location for keeping the model parameters.

    Returns
    -------
    Module:
        The model.
    """
    models = {
        # resnet
        'resnet50': resnet50,
        'resnet101': resnet101,
        'resnet152': resnet152,
        # resnest
        'resnest50': resnest50,
        'resnest101': resnest101,
        'resnest200': resnest200,
        'resnest269': resnest269,
        # resnet other variants
        'resnet50s': resnet50s,
        'resnet101s': resnet101s,
        'resnet152s': resnet152s,
        'resnet50d': resnet50d,
        'resnext50_32x4d': resnext50_32x4d,
        'resnext101_32x8d': resnext101_32x8d,
        # other segmentation backbones
        'xception65': xception65,
        'wideresnet38': wideresnet38,
        'wideresnet50': wideresnet50,
        # deepten paper
        'deepten_resnet50_minc': get_deepten_resnet50_minc,
        # segmentation models
        'fcn_resnet50_pcontext': get_fcn_resnet50_pcontext,
        'encnet_resnet50_pcontext': get_encnet_resnet50_pcontext,
        'encnet_resnet101_pcontext': get_encnet_resnet101_pcontext,
        'encnet_resnet50_ade': get_encnet_resnet50_ade,
        'encnet_resnet101_ade': get_encnet_resnet101_ade,
        'fcn_resnet50_ade': get_fcn_resnet50_ade,
        'psp_resnet50_ade': get_psp_resnet50_ade,
        'deeplab_resnest50_ade': get_deeplab_resnest50_ade,
        'deeplab_resnest101_ade': get_deeplab_resnest101_ade,
        }
    name = name.lower()
    if name not in models:
        raise ValueError('%s\n\t%s' % (str(name), '\n\t'.join(sorted(models.keys()))))
    net = models[name](**kwargs)
    return net
