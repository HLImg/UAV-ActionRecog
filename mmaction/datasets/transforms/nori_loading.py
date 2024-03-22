# -*- coding: utf-8 -*-
# @Time : 2024/03/22 13:09
# @Author : Liang Hao
# @FileName : nori_loading.py
# @Email : lianghao@whu.edu.cn

import mmcv
import numpy as np

from mmaction.registry import TRANSFORMS
from mmcv.transforms import BaseTransform


class RawFraneDecodeNoir2(BaseTransform):
    def __init__(self) -> None:
        super().__init__()