from .model_zoo import get_model
from .model_store import get_model_file
from .resnet import *
from .cifarresnet import *
from .base import *
from .fcn import *
from .psp import *
from .encnet import *
from .deeplab import *

def get_segmentation_model(name, **kwargs):
    from .fcn import get_fcn
    models = {
        'fcn': get_fcn,
        'psp': get_psp,
        'atten': get_atten,
        'encnet': get_encnet,
        'encnetv2': get_encnetv2,
        'deeplab': get_deeplab,
    }
    return models[name.lower()](**kwargs)
