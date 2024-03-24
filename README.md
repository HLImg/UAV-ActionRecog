# UAV-ActionRecog

An Introduction to Video Action Recognition from the Perspective of UAVs and Drones.

## Datasets

| Dataset                                                    | Year |            Modalities             | Environment | Frames | Classes | Resolution (RGB)  |                      Benchmark (Aerial)                      |
| :--------------------------------------------------------- | :--: | :-------------------------------: | :---------: | :----: | :-----: | :---------------: | :----------------------------------------------------------: |
| [VIRAT](https://viratdata.org/#getting-data)               | 2011 |                RGB                |   Outdoor   |  Many  |   24    |      Varying      |                              ~                               |
| [Okutama-Action](http://okutama-action.org/)               | 2017 |                RGB                |   Outdoor   |  70k   |   13    | $3840\times 2160$ | [SOTA-2023-75.94](https://paperswithcode.com/sota/action-recognition-on-okutama-action) |
| [UAV-GESTURE](https://github.com/asankagp/UAV-GESTURE)     | 2019 |                RGB                |   Outdoor   | 37.2k  |   13    | $1920\times 1080$ |                              ~                               |
| [Drone-Action](https://asankagp.github.io/droneaction/)    | 2019 |                RGB                |   Outdoor   | 66.9k  |   13    | $1920\times 1080$ | [SOTA-2023-95.9](https://paperswithcode.com/sota/action-recognition-on-drone-action) |
| [NEC-Drone](https://github.com/jinwchoi/NEC-Drone-Dataset) | 2020 |                RGB                |   Indoor    |        |   13    |        HD         |                                                              |
| [UAV-Human](https://github.com/SUTDCV/UAV-Human)           | 2021 | RGB, Depth, IR, Fisheye, Sketeton |   Outdoor   | 67.4k  |   155   | $1920\times 1080$ | [SOTA-2023-55.0](https://paperswithcode.com/sota/action-recognition-on-uav-human) |
| [RoCoG-v2](https://github.com/reddyav1/RoCoG-v2)           | 2023 |                RGB                |   Outdoor   |  107K  |    7    |         ~         | [SOTA-2023-40.2](https://paperswithcode.com/sota/action-recognition-on-rocog-v2) |
|                                                            |      |                                   |             |        |         |                   |                                                              |

1. **VIRAT**: 550个**低分辨率**视频，由静态和移动摄像机（称之为VIRAT地面和空中数据集）记录，涵盖23种户外活动类型。[基于VIRAT地面数据集的低分辨率行为识别挑战与研究](https://github.com/UgurDemir/Tiny-VIRAT)
2. **Okutama-Action**：使用无人机在棒球场上收集，数据集存在突发的摄像机运动，且摄像机90都仰角在视频中会产生严重的自遮挡和视角扭曲
3. **UAV-GESTURE**：低空悬停无人机记录的手势数据集，具有119个RGB视频
4. **Drone-Action**：240个高清视频片段，共计66919帧，所有视频都是在低空低速下拍摄的
5. **UAV-Human**：类别丰富，提供了多模态的数据
6. **RoCoG-v2**：由7个手势类的真实和合成视频组成，接近107K的合成视频基于Unity渲染。真实视频由两个户外地点收集，无人机在距离地面10米的地空中记录。

### UAVHuman DataSet

| ![image-20240117141536748](https://qiniu.lianghao.work/image-20240117141536748.png) | ![image-20240117141548549](https://qiniu.lianghao.work/image-20240117141548549.png) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |


## Paper Reading

- [x] [FAR: Fourier Aerial Video Recognition](https://github.com/divyakraman/ECCV2022_FARFourierAerialVideoRecognition) - [train_uavhuman_rgb_yaml](./docs/far_rgb.yml)
- [x] [X3D: Expanding Architectures for Efficient Video Recognition](https://arxiv.org/abs/2004.04730)
- [x] [PMI Sampler: Patch Similarity Guided Frame Selection For Aerial Action Recognition](https://lianghao.work/archives/pmisamplerpatchsimilarityguidedframeselection)
- [x] [MITFAS: Mutual Information based Temporal Feature Alignment and Sampling for Aerial Video Acton Recognition](https://lianghao.work/archives/mitfasmutualinformationbasedtemporalfeaturealignmentandsamplingforaerialvideoactonrecognition) 
- [ ] [Unsupervised and Semi-Supervised Domain Adaptation for Action Recognition from Drones](https://openaccess.thecvf.com/content_WACV_2020/papers/Choi_Unsupervised_and_Semi-Supervised_Domain_Adaptation_for_Action_Recognition_from_Drones_WACV_2020_paper.pdf)
- [ ] [InternVideo: General Video Foundation Models via Generative and Discriminative Learning](https://github.com/opengvlab/internvideo)

## The structure of PicToRestore

```shell
accelerate launch --config_file=tools/single_acc.yml --num_processes=8 main.py --config config/sidd_nafnet_wf32.yml --eval_ddp True --verbose True --train  # num_processes表示单机多卡训练时GPU个数，verbose表示是否在终端现实日志
```

```python
├── main.py # 程序启动文件
├── src
|   ├── train.py # 训练文件，一般不做改动
|   ├── arches # 网络结构
│   │   ├── __init__.py
|   ├── datasets # 数据类
│   │   ├── __init__.py
|   ├── loss # 损失函数
│   │   ├── __init__.py
|   ├── metrics # 评价指标
│   │   ├── __init__.py
|   ├── models # 根据config.yml初始化训练时所需的元素，并自定义网络训练时的操作
│   │   ├── __init__.py
|   ├── utils # 提供各种工具，主要包含上述模块的注册器
│   │   ├── __init__.py
```

## 使用笔记

### 模型复现

- [x] [C3D-UCF101](./usage/c3d_复现.md)

### 数据集处理

- [x] [UAVHuman](./usage/uavhuman_数据.md)

### Nori2数据

- [x] [UAVHuman](./usage/nori2_video.md)
- [x] [UCF101](./usage/nori2_video.md)
- [ ] [K400]()

## Thanks

部分代码和组织结构参考[BasicSR](https://github.com/XPixelGroup/BasicSR), [MMAction2](https://github.com/open-mmlab/mmaction2)以及其他卓越的工作。

