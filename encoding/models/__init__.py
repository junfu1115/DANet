from .model_zoo import get_model
from .base import *
from .fcn import *
from .psp import *
from .encnet import *

def get_segmentation_model(name, **kwargs):
    from .fcn import get_fcn
    models = {
        'fcn': get_fcn,
        'psp': get_psp,
        'encnet': get_encnet,
    }
    return models[name.lower()](**kwargs)
