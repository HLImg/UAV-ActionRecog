# -*- coding: utf-8 -*-
# @Time : 2024/03/21 19:28
# @Author : Liang Hao
# @FileName : slowfast_r50_train_ucf101_rgb_video.py
# @Email : lianghao@whu.edu.cn

_base_ = [
    './slowfast_r50_ucf101.py',
    '../default_runtime.py'
]

# dataset settings

dataset_type = 'RawframeDataset'
data_root = '/data/dataset/ucf101/rawframes'
data_root_val = '/data/dataset/ucf101/rawframes'

split = 1
ann_file_train = f'/data/dataset/ucf101/ucf101_train_split_{split}_rawframes.txt'
ann_file_test = f'/data/dataset/ucf101/ucf101_val_split_{split}_rawframes.txt'
ann_file_val = f'/data/dataset/ucf101/ucf101_val_split_{split}_rawframes.txt'

file_client_args = dict(io_backend='disk')

# dataset pipeline and dataloader
train_pipeline = [
    # 将视频转成帧，如果直接用帧则注释
    # dict(type='DecordInit', **file_client_args),
    dict(type='SampleFrames', clip_len=32, frame_interval=2, num_clips=1),
    # dict(type='DecordDecode'),
    dict(type='RawFrameDecode', **file_client_args),
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
    dict(type='RawFrameDecode', **file_client_args),
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
        num_clips=10,
        test_mode=True),
    # dict(type='DecordDecode'),
    dict(type='RawFrameDecode', **file_client_args),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='ThreeCrop', crop_size=256),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

train_dataloader = dict(
    batch_size=8,
    num_workers=8,
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
    batch_size=8,
    num_workers=8,
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
    num_workers=8,
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

# evaluator and Loop config
val_evaluator = dict(type='AccMetric')
test_evaluator = dict(type='AccMetric')


train_cfg = dict(
    type='EpochBasedTrainLoop', 
    max_epochs=256, 
    val_begin=1, 
    val_interval=5
)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# optimizer and scheduler

optim_wrapper = dict(
    optimizer = dict(
        type = 'SGD',
        lr = 0.1, 
        momentum = 0.9, 
        weight_decay = 1e-4
    ),
    
    clip_grad = dict(
        max_norm = 40,
        norm_type = 2
    )
)

param_scheduler = [
    dict(
        type = 'LinearLR',
        start_factor = 0.1,
        by_epoch = True,
        begin = 0,
        end = 34,
        convert_to_iter_based = True
    ),
    dict(
        type='CosineAnnealingLR',
        T_max = 256,
        eta_min = 0,
        by_epoch = True,
        begin = 0,
        end = 256
    )
]


default_hooks = dict(
    checkpoint=dict(
        interval=4, 
        max_keep_ckpts=3
    ), 
    logger=dict(
        interval=100
    )
)
