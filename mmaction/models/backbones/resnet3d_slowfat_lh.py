# -*- coding: utf-8 -*-
# @Time : 2024/03/20 10:53
# @Author : Liang Hao
# @FileName : resnet3d_slowfat_lh.py
# @Email : lianghao@whu.edu.cn

# Copyright (c) OpenMMLab. All rights reserved.


import warnings

from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import ConvModule
from mmengine.logging import MMLogger, print_log
from mmengine.model import BaseModule
from mmengine.model.weight_init import kaiming_init
from mmengine.runner.checkpoint import _load_checkpoint, load_checkpoint

from .resnet3d import ResNet3d
from mmaction.registry import MODELS


class DeConvModule(BaseModule):
    """
    3D Transposed Conv的封装, 添加了bn归一化层和relu激活层
    """
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size = 3,
                 stride = (1, 1, 1),
                 padding = 0,
                 bias = False,
                 with_bn = True,
                 with_relu = True
                 ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.with_bn = with_bn
        self.with_relu = with_relu
        
        self.conv = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        )
        
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        assert len(x.shape) == 6
        
        N, C, T, H, W = x.shape
        out_shape = (N, self.out_channels, self.stride[0] * T,
                     self.stride[1] * H, self.stride[2] * W)
        
        x = self.conv(x, output_size=out_shape)
        
        if self.with_bn:
            x = self.bn(x)
        
        if self.with_relu:
            x = self.relu(x)



class ResNet3dPathway(ResNet3d):
    def __init__(self,
                 lateral = False,
                 lateral_inv = False,
                 lateral_norm = False,
                 speed_ratio = 8,
                 channel_ratio = 8,
                 fusion_kernel = 5,
                 lateral_infl = 2,
                 lateral_activate = [1, 1, 1, 1],
                 **kwargs):
        """A pathway of Slowfast based on ResNet3d.
        Args:
            lateral (bool, optional): 决定是否启用横向连接
            lateral_inv (bool, optional): 是否使用deconv来提升来自另一条路径特征的时间维度
            lateral_norm (bool, optional): 在横向连接层中是否归一化
            speed_ratio (int, optional): 快慢路径在时间维度上的速度率
            channel_ratio (int, optional): 减少快路径通道数目的比率
            fusion_kernel (int, optional): 横向融合的卷积核大小
            lateral_infl (int, optional): 膨胀通道的比率
            lateral_activate (list, optional): 激活横向连接的标志
        """
        self.lateral = lateral
        self.lateral_inv = lateral_inv
        self.lateral_norm = lateral_norm
        self.speed_ratio = speed_ratio
        self.channel_ratio = channel_ratio
        self.fusion_kernel = fusion_kernel
        self.lateral_infl = lateral_infl
        self.lateral_activate = lateral_activate
        self._calculate_lateral_inplanes(kwargs)

        super().__init__(**kwargs)
        self.inplanes = self.base_channels
        if self.lateral and self.lateral_activate[0] == 1:
            if self.lateral_inv:
                self.conv1_lateral = DeConvModule(
                    self.inplanes * self.channel_ratio,
                    self.inplanes * self.channel_ratio // lateral_infl,
                    kernel_size=(fusion_kernel, 1, 1),
                    stride=(self.speed_ratio, 1, 1),
                    padding=((fusion_kernel - 1) // 2, 0, 0),
                    with_bn=True,
                    with_relu=True)
            else:
                self.conv1_lateral = ConvModule(
                    self.inplanes // self.channel_ratio,
                    self.inplanes * lateral_infl // self.channel_ratio,
                    kernel_size=(fusion_kernel, 1, 1),
                    stride=(self.speed_ratio, 1, 1),
                    padding=((fusion_kernel - 1) // 2, 0, 0),
                    bias=False,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg if self.lateral_norm else None,
                    act_cfg=self.act_cfg if self.lateral_norm else None)

        self.lateral_connections = []
        for i in range(len(self.stage_blocks)):
            planes = self.base_channels * 2**i
            self.inplanes = planes * self.block.expansion

            if lateral and i != self.num_stages - 1 \
                    and self.lateral_activate[i + 1]:
                # no lateral connection needed in final stage
                lateral_name = f'layer{(i + 1)}_lateral'
                if self.lateral_inv:
                    conv_module = DeConvModule(
                        self.inplanes * self.channel_ratio,
                        self.inplanes * self.channel_ratio // lateral_infl,
                        kernel_size=(fusion_kernel, 1, 1),
                        stride=(self.speed_ratio, 1, 1),
                        padding=((fusion_kernel - 1) // 2, 0, 0),
                        bias=False,
                        with_bn=True,
                        with_relu=True)
                else:
                    conv_module = ConvModule(
                        self.inplanes // self.channel_ratio,
                        self.inplanes * lateral_infl // self.channel_ratio,
                        kernel_size=(fusion_kernel, 1, 1),
                        stride=(self.speed_ratio, 1, 1),
                        padding=((fusion_kernel - 1) // 2, 0, 0),
                        bias=False,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg if self.lateral_norm else None,
                        act_cfg=self.act_cfg if self.lateral_norm else None)
                setattr(self, lateral_name, conv_module)
                self.lateral_connections.append(lateral_name)
        
    
    def _calculate_lateral_inplanes(self, kwargs):
        depth = kwargs.get('depth', 50)
        expansion = 1 if depth < 50 else 4
        base_channels = kwargs.get('base_channels', 64)
        lateral_inplanes = []
        
        for i in range(kwargs.get('num_stages', 4)):
            if expansion % 2 == 0:
                planes = base_channels * (2 ** i) * ((expansion // 2) ** (i > 0))
            else:
                planes = base_channels * (2**i) // (2**(i > 0))
            
            if self.lateral and self.lateral_activate[i]:
                if self.lateral_inv:
                    # 横向连接
                    lateral_inplane = planes * self.channel_ratio // self.lateral_infl
                else:
                    lateral_inplane = planes * self.lateral_infl // self.channel_ratio
            else:
                lateral_inplane = 0
                
            lateral_inplanes.append(lateral_inplane)
        
        self.lateral_inplanes = lateral_inplanes
        
    
    def inflate_weights(self, logger: MMLogger) -> None:
        """将resnet2d参数膨胀到resnet3d
        resnet3d 和 resnet2d 的区别主要在卷积核多了处理时序的维度。为了利用
        2D模型中预训练的参数，conv2d模型的权重应该膨胀以适应以conv3d。
        路径中的横向连接部分不从2d权重膨胀
        """
        # 加载resnet2d的权重
        state_dict_r2d = _load_checkpoint(self.pretrained, map_location='cpu')
        if 'state_dict' in state_dict_r2d:
            state_dict_r2d = state_dict_r2d['state_dict']
        
        inflated_param_names = []
        
        for name, module in self.named_modules():
            if 'lateral' in name:
                continue
            if isinstance(module, 'ConvModule'):
                # The role of ConvModule is to wrap conv+bn+relu layers
                if 'downsample' in name:
                    # layer{X}.{Y}.downsample.conv->layer{X}.{Y}.downsample.0
                    original_conv_name = name + '.0'
                    # layer{X}.{Y}.downsample.bn->layer{X}.{Y}.downsample.1
                    original_bn_name = name + '.1'
                else:
                    # layer{X}.{Y}.conv{n}.conv->layer{X}.{Y}.conv{n}
                    original_conv_name = name
                    # layer{X}.{Y}.conv{n}.bn->layer{X}.{Y}.bn{n}
                    original_bn_name = name.replace('conv', 'bn')
                
                if original_conv_name + '.weight' not in state_dict_r2d:
                    logger.warning(f'Module not exist in the state_dict_r2d'
                                   f': {original_conv_name}')
                else:
                    self._inflate_conv_params(module.conv, state_dict_r2d,
                                              original_conv_name,
                                              inflated_param_names)
                
                if original_bn_name + '.weight' not in state_dict_r2d:
                    logger.warning(f'Module not exist in the state_dict_r2d'
                                   f': {original_bn_name}')
                
                else:
                    self._inflate_bn_params(module.bn, state_dict_r2d,
                                            original_bn_name,
                                            inflated_param_names)
                
        # check if any parameters in the 2d checkpoint are not loaded
        remaining_names = set(
            state_dict_r2d.keys()) - set(inflated_param_names)
        if remaining_names:
            logger.info(f'These parameters in the 2d checkpoint are not loaded'
                        f': {remaining_names}')
        
    
    def _inflate_conv_params(self, 
                             conv3d, 
                             state_dict_2d, 
                             module_name_2d, 
                             inflated_param_names):
        """
        将2d conv module 膨胀到 3d
        """
        weight_2d_name = module_name_2d + '.weight'
        conv2d_weight = state_dict_2d[weight_2d_name]
        # out_ch x in_ch x kh x kw
        old_shape = conv2d_weight.shape
        # out_ch x in_ch x kt x kh x kw
        new_shape = conv3d.weight.data.shape
        kernel_t = new_shape[2]
        
        if new_shape[1] != old_shape[1]:
            if new_shape[1] < old_shape[1]:
                warnings.warn(f'The parameter of {module_name_2d} is not'
                              'loaded due to incompatible shapes. ')
                return
            
            # 当3D卷积的输入维度 大于 2D卷积时，对超出的部分进行0填充
            new_channels = new_shape[1] - old_shape[1]
            pad_shape = old_shape
            pad_shape = pad_shape[:1] + (new_channels, ) + pad_shape[2:]
            
            conv2d_weight = torch.cat(
                (conv2d_weight,
                 torch.zeros(pad_shape).type_as(conv2d_weight).to(
                     conv2d_weight.device)),
                dim=1)
        
        new_weight = conv2d_weight.data.unsqueeze(2).expand_as(conv3d.weight) / kernel_t
        conv3d.weight.data.copy_(new_weight)
        inflated_param_names.append(weight_2d_name)
        
        if getattr(conv3d, 'bias') is not None:
            bias_2d_name = module_name_2d + '.bias'
            conv3d.bias.data.copy_(state_dict_2d[bias_2d_name])
            inflated_param_names.append(bias_2d_name)
    
    def _freeze_stages(self) -> None:
        """Prevent all the parameters from being optimized before
        `self.frozen_stages`."""
        if self.frozen_stages >= 0:
            self.conv1.eval()
            for param in self.conv1.parameters():
                param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

            if i != len(self.res_layers) and self.lateral:
                # No fusion needed in the final stage
                lateral_name = self.lateral_connections[i - 1]
                conv_lateral = getattr(self, lateral_name)
                conv_lateral.eval()
                for param in conv_lateral.parameters():
                    param.requires_grad = False
    
    def init_weights(self, pretrained: Optional[str] = None) -> None:
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        if pretrained:
            self.pretrained = pretrained

        # Override the init_weights of i3d
        super().init_weights()
        for module_name in self.lateral_connections:
            layer = getattr(self, module_name)
            for m in layer.modules():
                if isinstance(m, (nn.Conv3d, nn.Conv2d)):
                    kaiming_init(m)
                    

pathway_cfg = {
    'resnet3d': ResNet3dPathway,
    # TODO: BNInceptionPathway
}

def build_pathway(cfg, *args, **kwargs):
    if not (isinstance(cfg, dict) and 'type' in cfg):
        raise TypeError('cfg must be a dict containing the key "type"')
    
    cfg_ = cfg.copy()
    pathway_type = cfg_.pop('type')
    
    if pathway_type not in pathway_cfg:
        raise KeyError(f'Unrecognized pathway type {pathway_type}')
    
    pathway_cls = pathway_cfg[pathway_type]
    pathway = pathway_cls(*args, **kwargs, **cfg_)
    
    return pathway


@MODELS.register_module()
class ResNet3dSlowFastLh(BaseModule):
    def __init__(self, 
                 pretrained = None,
                 resample_rate = 8,
                 speed_ratio = 8,
                 channel_ratio = 8,
                 slow_pathway = dict(
                     type='resnet3d',
                     depth=50,
                     pretrained=None,
                     lateral=True,
                     conv1_kernel=(1, 7, 7),
                     conv1_stride_t=1,
                     pool1_stride_t=1,
                     inflate=(0, 0, 1, 1)),
                 fast_pathway = dict(
                     type='resnet3d',
                     depth=50,
                     pretrained=None,
                     lateral=False,
                     base_channels=8,
                     conv1_kernel=(5, 7, 7),
                     conv1_stride_t=1,
                     pool1_stride_t=1),
                 init_cfg = None
                 ):
        super().__init__(init_cfg=init_cfg)
        
        self.pretrained = pretrained
        self.resample_rate = resample_rate
        self.speed_ratio = speed_ratio
        self.channel_ratio = channel_ratio
        
        if slow_pathway['lateral']:
            slow_pathway['speed_ratio'] = speed_ratio
            slow_pathway['channel_ratio'] = channel_ratio
        
        self.slow_path = build_pathway(slow_pathway)
        self.fast_path = build_pathway(fast_pathway)
    
    def init_weights(self, pretrained: Optional[str] = None) -> None:
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        if pretrained:
            self.pretrained = pretrained

        if isinstance(self.pretrained, str):
            logger = MMLogger.get_current_instance()
            msg = f'load model from: {self.pretrained}'
            print_log(msg, logger=logger)
            # Directly load 3D model.
            load_checkpoint(self, self.pretrained, strict=True, logger=logger)
        elif self.pretrained is None:
            # Init two branch separately.
            self.fast_path.init_weights()
            self.slow_path.init_weights()
        else:
            raise TypeError('pretrained must be a str or None')
    
    
    def forward(self, x):
        x_slow = F.interpolate(x, 
                               mode='nearest', 
                               scale_factor=(1.0 / self.resample_rate, 1.0, 1.0))
        
        x_slow = self.slow_path.conv1(x_slow)
        x_slow = self.slow_path.maxpool(x_slow)
        
        x_fast = F.interpolate(x, 
                               mode='nearest',
                               scale_factor=(1.0 / (self.resample_rate // self.speed_ratio), 1.0, 1.0))
        x_fast = self.fast_path.conv1(x_fast)
        x_fast = self.fast_path.maxpool(x_fast)
        
        if self.slow_path.lateral:
             x_fast_lateral = self.slow_path.conv1_lateral(x_fast)
             x_slow = torch.cat((x_slow, x_fast_lateral), dim=1)
        
        for i, layer_name in enumerate(self.slow_path.res_layers):
            res_layer = getattr(self.slow_path, layer_name)
            x_slow = res_layer(x_slow)
            res_layer_fast = getattr(self.fast_path, layer_name)
            x_fast = res_layer_fast(x_fast)
            
            if (i != len(self.slow_path.res_layers) - 1
                    and self.slow_path.lateral):
                lateral_name = self.slow_path.lateral_connections[i]
                conv_lateral = getattr(self.slow_path, lateral_name)
                x_fast_lateral = conv_lateral(x_fast)
                x_slow = torch.cat((x_slow, x_fast_lateral), dim=1)
        
        out = (x_slow, x_fast)
        return out