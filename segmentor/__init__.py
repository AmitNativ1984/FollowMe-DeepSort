from .segmentor import Segmentor

__all__ = ['build_segmentor']

def build_segmentor(cfg, cls_dict):
    return Segmentor(cfg, cls_dict)
