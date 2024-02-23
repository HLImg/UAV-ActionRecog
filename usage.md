## C3D-UCF101复现
### 1. 数据准备
ucf101数据集路径如下
```shell
├── videos/
    ├── ApplyEyeMakeup/
    ├── ...
├── classInd.txt
├── testlist01.txt
├── testlist02.txt
├── testlist03.txt
├── trainlist01.txt
├── trainlist02.txt
└── trainlist03.txt
```
1. **抽取视频帧**：
    ```shell
    # 进入mmaction2的数据工具类
    cd mmaction2/tools/data
    # 根据数据集的具体路径和指定shape使用opencv抽取视频帧
    python build_rawframes.py /data/dataset/ucf101/videos/ /data/dataset/ucf101/rawframes/ --task rgb --level 2 --ext avi --new-width 320 --new-height 240 --use-opencv
    ```
2. **生成list文件**：**如果是新的数据集，需要在[parse_file_list.py](./tools/data/parse_file_list.py)自定义该数据集的划分函数，并在[build_file_list.py](./tools/data/build_file_list.py)中导入该数据集相关的函数**

    ```shell
    # 在当前项目目录下(mmaction2/)
    PYTHONPATH=. python tools/data/build_file_list.py ucf101 /data/dataset/ucf101/rawframes --level 2 --format rawframes --shuffle
    ```

    最终结果会保存到[./data/ucf101/](./data/ucf101/)下面，注意运行之前相关的注释文件被默认保留在该目录下，如果需要指定注释文件的存储位置，那么修改[parse_file_list.py](./tools/data/parse_file_list.py)中的代码。

    最终的文件目录为
    ```
    data
    └── ucf101
        ├── annotations
        │   ├── classInd.txt
        │   ├── testlist01.txt
        │   ├── testlist02.txt
        │   ├── testlist03.txt
        │   ├── trainlist01.txt
        │   ├── trainlist02.txt
        │   └── trainlist03.txt
        ├── ucf101_train_split_1_rawframes.txt
        ├── ucf101_train_split_2_rawframes.txt
        ├── ucf101_train_split_3_rawframes.txt
        ├── ucf101_val_split_1_rawframes.txt
        ├── ucf101_val_split_2_rawframes.txt
        └── ucf101_val_split_3_rawframes.txt
    ```

### 2. 准备C3D的config

在自己的workdir中准备配置文件
```shell
mkdir ./work_dir
```

根据自己的具体情况进行配置，[c3d_pretrained_ucf101_rgb.py](./work_dir/c3d/c3d_pretrained_ucf101_rgb.py)和[c3d_sports1m_pretrained_model.py](./work_dir/c3d/c3d_sports1m_pretrained_model.py)

### 3. 训练

```shell
# 分布式训练
bash tools/dist_train.sh work_dir/c3d/c3d_pretrained_ucf101_rgb.py 8 --work-dir results/checkpoints --seed 0 --deterministic
# 
python tools/train.py configs/recognition/c3d/c3d_sports1m-pretrained_8xb30-16x1x1-45e_ucf101-rgb.py --seed=0 --deterministic
```

1. 在**config**文件中，[default_runtimes.pt](./work_dir/default_runtime.py)是必需的，否则会存在**Recognizer3D没有被注册的错误**
    ```python
            _base_ = [
            './c3d_sports1m_pretrained_model.py',
            '../default_runtime.py'
        ]
    ```

https://blog.csdn.net/qq_29761909/article/details/126727152

