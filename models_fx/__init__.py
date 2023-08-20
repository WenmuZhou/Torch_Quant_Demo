# -*- coding: utf-8 -*-
# @Time    : 2023/8/19 19:37
# @Author  : zhoujun
from torchvision import models


def build_model(name, *args, **kwargs):
    assert hasattr(models, name), f"{name} is not in {dir(models)}"
    return getattr(models, name)(*args, **kwargs)
