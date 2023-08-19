# -*- coding: utf-8 -*-
# @Time    : 2023/8/19 19:37
# @Author  : zhoujun

from .mobilenetv2 import MobileNetV2
from .resnet import resnet18, resnet34, resnet50, resnet101, resnet152, resnext50_32x4d, resnext101_32x8d, \
    wide_resnet50_2, wide_resnet101_2

__all__ = [
    'MobileNetV2',
    'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
    'wide_resnet50_2', 'wide_resnet101_2',
]


def build_model(name, *args, **kwargs):
    assert name in __all__, print(f'{name} is not in {__all__}')
    return eval(name)(*args, **kwargs)