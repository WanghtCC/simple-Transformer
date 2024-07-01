from __future__ import absolute_import

from .vit_model import ViT

__factory = {
    'vit': ViT,
}

def get_names():
    return __factory.keys()

def init_model(name, *args, **kwargs):
    if name not in __factory.keys():
        print(__factory.keys())
        raise KeyError("Unknown model:", name)
    return __factory[name](*args, **kwargs)