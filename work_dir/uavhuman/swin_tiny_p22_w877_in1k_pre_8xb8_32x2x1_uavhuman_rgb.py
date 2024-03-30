# -*- coding: utf-8 -*-
# @Time : 2024/03/26 22:11
# @Author : Liang Hao
# @FileName : swin_tiny_p22_w877_in1k_pre_8xb8_32x2x1_uavhuman_rgb.py
# @Email : lianghao@whu.edu.cn

_base_ = [
    './model/swin_tiny.py',
    '../default_runtime.py'
]

model = dict(
    backbone=dict(
        pretrained=  # noqa: E251
        'swin-tiny-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb_20220930-241016b2.pth'  # noqa: E501
    ))


dataset_type = 'RawframeDataset'
data_root = '/data/dataset/uavhuman/rawframes'
data_root_val = '/data/dataset/uavhuman/rawframes'
split = 2  # official train/test splits. valid numbers: 1, 2, 3
ann_file_train = f'data/uavhuman/uavhuman_train_split_{split}_rawframes.txt'
ann_file_val = f'data/uavhuman/uavhuman_val_split_{split}_rawframes.txt'
ann_file_test = f'data/uavhuman/uavhuman_val_split_{split}_rawframes.txt'

file_client_args_train = dict(
    io_backend='disk',
    nori_file = 'data/uavhuman/uavhuman_train_split_1_nid.json',
    dtype = 'uint8',
    retry = 60
)

file_client_args_eval = dict(
    io_backend='disk',
    nori_file = 'data/uavhuman/uavhuman_val_split_1_nid.json',
    dtype = 'uint8',
    retry = 60
)

train_pipeline = [
    # dict(type='DecordInit', **file_client_args),
    dict(type='SampleFrames', clip_len=32, frame_interval=2, num_clips=1),
    # dict(type='DecordDecode'),
    dict(type='RawFrameDecodeNoir2', **file_client_args_train),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

val_pipeline = [
    # dict(type='DecordInit', **file_client_args),
    dict(
        type='SampleFrames',
        clip_len=32,
        frame_interval=2,
        num_clips=1,
        test_mode=True),
    # dict(type='DecordDecode'),
    dict(type='RawFrameDecodeNoir2', **file_client_args_eval),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

test_pipeline = [
    # dict(type='DecordInit', **file_client_args),
    dict(
        type='SampleFrames',
        clip_len=32,
        frame_interval=2,
        num_clips=4,
        test_mode=True),
    # dict(type='DecordDecode'),
    dict(type='RawFrameDecodeNoir2', **file_client_args_eval),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='ThreeCrop', crop_size=256),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]


train_dataloader = dict(
    batch_size=3,
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
    batch_size=3,
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


val_evaluator = dict(type='AccMetric')
test_evaluator = val_evaluator

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=99, val_begin=1, val_interval=3)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(
        type='AdamW', lr=1e-3, betas=(0.9, 0.999), weight_decay=0.02),
    constructor='SwinOptimWrapperConstructor',
    paramwise_cfg=dict(
        absolute_pos_embed=dict(decay_mult=0.),
        relative_position_bias_table=dict(decay_mult=0.),
        norm=dict(decay_mult=0.),
        backbone=dict(lr_mult=0.1)))

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.1,
        by_epoch=True,
        begin=0,
        end=2.5,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=99,
        eta_min=0,
        by_epoch=True,
        begin=0,
        end=99)
]

default_hooks = dict(
    checkpoint=dict(interval=3, max_keep_ckpts=5), logger=dict(interval=100))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (8 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=64)