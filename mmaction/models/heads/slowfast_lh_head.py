# -*- coding: utf-8 -*-
# @Time : 2024/03/20 13:24
# @Author : Liang Hao
# @FileName : slowfast_lh_head.py
# @Email : lianghao@whu.edu.cn

# Copyright (c) OpenMMLab. All rights reserved.'

import torch
from mmengine.model.weight_init import normal_init
from torch import Tensor, nn

from mmaction.registry import MODELS
from mmaction.utils import ConfigType
from .base import BaseHead

@MODELS.register_module()
class SlowFastLHead(BaseHead):
    """The classification head for SlowFast.

    Args:
            num_classes (_type_): 类别数目
            in_channels (_type_): 输入特征的通道数量
            loss_cls (_type_, optional): 用于构建损失函数的配置
            spatial_type (str, optional): 在空间上进行池化的类型，均值池化或最大值池化
            dropout_ratio (float, optional): Probability of dropout layer. Default: 0.8.
            init_std (float, optional): Std value for Initiation. Default: 0.01.
    """
    def __init__(self, 
                 num_classes,
                 in_channels, 
                 loss_cls = dict(type = 'CrossEntropyLoss'),
                 spatial_type = 'avg',
                 dropout_ratio = 0.8,
                 init_std = 0.01,
                 **kwargs):
        super().__init__(num_classes, in_channels, loss_cls, **kwargs)
        
        self.spatial_type = spatial_type
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std
        
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        
        self.fc_cls = nn.Linear(in_channels, num_classes)
        
        if self.spatial_type == 'avg':
            self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        else:
            self.avg_pool = None
    
    
    def init_weights(self):
        normal_init(self.fc_cls, std=self.init_std)
        
    def forward(self, x, **kwargs):
        x_slow, x_fast = x
        x_slow = self.avg_pool(x_slow)
        x_fast = self.avg_pool(x_fast)
        
        x = torch.cat((x_fast, x_slow), dim=1)
        
        if self.dropout is not None:
            x = self.dropout(x)
        
        x = x.view(x.size(0), -1)
        cls_score = self.fc_cls(x)
        
        return cls_score