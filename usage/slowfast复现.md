## SlowFast-UCF101
虽然mmaction2已经提供了详细的配置问题，但为了增强对mmaction2的熟悉程度，将按照已有的代码手动写（copy）一下，重点掌握slowfast模型结果已经mmaction2框架的熟悉程度。为了与已有的SlowFast不冲突，在已有的名字上加上'_lh'。并将配置文件放在[work_dir目录下](./../work_dir/slowfast/)

### 数据集准备
ucf101数据集已经抽帧处理，[具体过程可见C3D_复现](./c3d_复现.md)

数据处理之后，在配置文件中初始化数据加载和处理的**Pipeline**，以及训练时使用的**DataLoader**

```python
train_pipeline = []
val_pipeline = []
test_pipeline = []

train_dataloader = dict()
val_dataloader = dict()
test_dataloader = dict()
```

### 网络模型
网路模型的搭建主要集中在[mmaction/models](../mmaction/models/)目录下修改。模型的配置文件主要为

```python
model = dict(
    type = 'Recognizer3D',
    backbone = dict(),
    cls_head = dict(),
    data_preprocessor = dict()
)
```

#### Backbone
在[resnet3d_slowfat_lh.py](../mmaction/models/backbones/resnet3d_slowfat_lh.py)中，基于resnet3d搭建slowfast的slow和fast路径，并将两个路径提出的特征进行输出。

#### CLS_Heads
在[slowfast_lh_head.p](../mmaction/models/heads/slowfast_lh_head.py)中将来自backbone提取的特征用于分类，最终输出分类score，形状为(batch size, num_classes)

#### Recognizer
[mmaction/models/recognizers](../mmaction/models/recognizers)

#### Data Preprocessor
[mmaction/models/data_preprocessors](../mmaction/models/data_preprocessors)


### 优化器和学习率调度器准备


```python
# 优化器
optim_wrapper = dict()
# 调度器
param_scheduler = []
```

### 评估器
```python
val_evaluator = dict(type='AccMetric')
test_evaluator = val_evaluator

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=256, val_begin=1, val_interval=5)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
```


### 训练

根据[配置文件](../work_dir/slowfast)执行如下的命令进行训练

```shell
bash tools/dist_train.sh work_dir/slowfast/slowfast_r50_train_ucf101_rgb_video.py 8 --work-dir results/slowfast_r50_ucf101 --seed=0 --deterministic
```