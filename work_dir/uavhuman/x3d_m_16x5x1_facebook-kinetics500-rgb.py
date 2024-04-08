# -*- coding: utf-8 -*-
# @Time : 2024/04/07 20:54
# @Author : Liang Hao
# @FileName : x3d_m_16x5x1_facebook-kinetics500-rgb.py
# @Email : lianghao@whu.edu.cn

_base_ = [
    './model/x3d.py',
    '../default_runtime.py'
]

model = dict(
    backbone=dict(
        pretrained="model_zoo/x3d_m_16x5x1_facebook-kinetics400-rgb_20201027-3f42382a.pth"
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
    dict(type='SampleFrames', clip_len=8, frame_interval=1, num_clips=1),
    # dict(type='DecordDecode'),
    dict(type='RawFrameDecodeNoir2', **file_client_args_train),
    dict(type='Resize', scale=(620, 620)),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(540, 540), keep_ratio=False),
    # dict(type='ThreeCrop', crop_size=540),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

val_pipeline = [
    # dict(type='DecordInit', **file_client_args),
    dict(
        type='SampleFrames',
        clip_len=8,
        frame_interval=1,
        num_clips=1,
        test_mode=True),
    # dict(type='DecordDecode'),
    dict(type='RawFrameDecodeNoir2', **file_client_args_eval),
    dict(type='Resize', scale=(540, 540), keep_ratio=False),
    # dict(type='ThreeCrop', crop_size=540),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

test_pipeline = [
    # dict(type='DecordInit', **file_client_args),
    dict(
        type='SampleFrames',
        clip_len=8,
        frame_interval=1,
        num_clips=1,
        test_mode=True),
    # dict(type='DecordDecode'),
    dict(type='RawFrameDecodeNoir2', **file_client_args_eval),
    dict(type='Resize', scale=(540, 540), keep_ratio=False),
    # dict(type='ThreeCrop', crop_size=540),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]


train_dataloader = dict(
    batch_size=3,
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
    batch_size=1,
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


val_evaluator = dict(type='AccMetric')
test_evaluator = val_evaluator

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=400, val_begin=1, val_interval=5)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

optim_wrapper = dict(
    optimizer=dict(
        type='SGD', lr=2e-5, momentum=0.9, weight_decay=5e-5),
    )

param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        T_max=400,
        eta_min=1e-7,
        by_epoch=True)
]

default_hooks = dict(
    checkpoint=dict(interval=5, max_keep_ckpts=5), logger=dict(interval=100))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (8 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=24)