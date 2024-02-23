# -*- encoding: utf-8 -*-
# @Author  :   Liang Hao 
# @Time    :   2024/02/22 21:30:50
# @File    :   c3d_pretrained_ucf101_rgb.py
# @Contact :   lianghao@whu.edu.cn


_base_ = [
    './c3d_sports1m_pretrained_model.py',
    '../default_runtime.py'
]

# dataset settings
# 使用抽取的帧而不是视频
dataset_type = 'RawframeDataset'
data_root = '/data/dataset/ucf101/rawframes'
data_root_val = '/data/dataset/ucf101/rawframes'
# 数据集划分方式，提供了3种划分方式
split = 1
ann_file_train = f'/data/dataset/ucf101/ucf101_train_split_{split}_rawframes.txt'
ann_file_test = f'/data/dataset/ucf101/ucf101_val_split_{split}_rawframes.txt'
ann_file_val = f'/data/dataset/ucf101/ucf101_val_split_{split}_rawframes.txt'

file_client_args = dict(io_backend='disk')

# dataset pipeline
train_pipeline = [
    # 将视频转成帧，如果直接用帧则注释
    # dict(type='DecordInit', **file_client_args),
    dict(type='SampleFrames', clip_len=16, frame_interval=1, num_clips=1),
    # dict(type='DecordDecode'),
    dict(type='RawFrameDecode', **file_client_args),
    dict(type='Resize', scale=(-1, 128)),
    dict(type='RandomCrop', size=112),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

val_pipeline = [
    # dict(type='DecordInit', **file_client_args),
    dict(
        type='SampleFrames',
        clip_len=16,
        frame_interval=1,
        num_clips=1,
        test_mode=True),
    # dict(type='DecordDecode'),
    dict(type='RawFrameDecode', **file_client_args),
    dict(type='Resize', scale=(-1, 128)),
    dict(type='CenterCrop', crop_size=112),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

test_pipeline = [
    # dict(type='DecordInit', **file_client_args),
    dict(
        type='SampleFrames',
        clip_len=16,
        frame_interval=1,
        num_clips=10,
        test_mode=True),
    # dict(type='DecordDecode'),
    dict(type='RawFrameDecode', **file_client_args),
    dict(type='Resize', scale=(-1, 128)),
    dict(type='CenterCrop', crop_size=112),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

# dataloader
train_dataloader = dict(
    batch_size=30,
    num_workers=4,
    # 数据加载完并不会关闭worker进程，而是保持现有的worker进程
    # 继续进行下一个Epoch的数据加载，加快训练速度，要求num_workers ≥ 1
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset = dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=dict(img=data_root),
        pipeline=train_pipeline
    )
)

val_dataloader = dict(
    batch_size=30,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=dict(img=data_root_val),
        pipeline=val_pipeline,
        test_mode=True
    )
)

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=dict(img=data_root_val),
        pipeline=test_pipeline,
        test_mode=True
    )
)

# evaluator
val_evaluator = dict(type='AccMetric')
test_evaluator = dict(type='AccMetric')

# Loop config

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=45, val_begin=1, val_interval=5)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# optimizer, scheduler, etc.

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=45,
        by_epoch=True,
        milestones=[20, 40],
        gamma=0.1
    )
]

optim_wrapper = dict(
    optimizer=dict(
        type='SGD',
        lr=0.001,
        momentum=0.9,
        weight_decay=0.0005
    ),
    clip_grad=dict(
        max_norm=40,
        norm_type=2
    )
)

default_hooks = dict(
    checkpoint=dict(interval=5)
)

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (30 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=240)
