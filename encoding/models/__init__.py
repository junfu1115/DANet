from .model_zoo import get_model
from .base import *
from .fcn import *
from .encnet import *

def get_segmentation_model(name, **kwargs):
    from .fcn import get_fcn
    models = {
        'fcn': get_fcn,
        'encnet': get_encnet,
    }
    return models[name.lower()](**kwargs)
