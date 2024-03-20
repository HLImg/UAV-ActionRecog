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
bash tools/dist_train.sh work_dir/c3d/c3d_pretrained_ucf101_rgb.py 8 --work-dir results/checkpoints --seed=0 --deterministic
# 
python tools/train.py work_dir/c3d_official/c3d_pretraied_ucf101_rgb_video.py --work-dir results/checkpoints --seed=0 --deterministic
```



1. 在**config**文件中，[default_runtimes.pt](./work_dir/default_runtime.py)是必需的，否则会存在**Recognizer3D没有被注册的错误**
    ```python
            _base_ = [
            './c3d_sports1m_pretrained_model.py',
            '../default_runtime.py'
        ]
    ```


## UAVHuman数据准备
### 1. 抽取视频帧

```shell
cd tools/data
conda activate mmaction
python build_rawframes.py /data/dataset/uavhuman/videos/ /data/dataset/uavhuman/rawframes/ --task rgb --level 2 --ext avi --use-opencv
```

**该数据抽帧所需存储空间较大**

### 2.生成list文件
在[parse_file_list.py](./tools/data/parse_file_list.py)中加入uavhuman数据集的splits函数
```python
def parse_uavhuman_splits(level):
    class_index_file = '/data/dataset/uavhuman/annotations/classInd.txt'
    train_file_template = '/data/dataset/uavhuman/annotations/trainlist{:02d}.txt'
    test_file_template = '/data/dataset/uavhuman/annotations/testlist{:02d}.txt'
    
    with open(class_index_file, 'r') as fin:
        class_index = [x.strip().split() for x in fin]
        
    class_mapping = {x[1]: int(x[0]) - 1 for x in class_index}
    
    def line_to_map(line):
        items = line.strip().split()
        video = osp.splitext(items[0])[0]
        if level == 1:
            video = osp.basename(video)
            label = items[0]
        elif level == 2:
            video = osp.join(
                osp.basename(osp.dirname(video)), osp.basename(video))
            label = class_mapping[osp.dirname(items[0])]
        return video, label
    
    splits = []
    # i 表示第i种划分方式
    for i in range(1, 2):
        with open(train_file_template.format(i), 'r') as fin:
            train_list = [line_to_map(x) for x in fin]

        with open(test_file_template.format(i), 'r') as fin:
            test_list = [line_to_map(x) for x in fin]
        splits.append((train_list, test_list))

    return splits
```

紧接着在[build_file_list.py](./tools/data/build_file_list.py)中导入**parse_uavhuman_splits**函数，并在**main函数**中添加下面语句
```python
# 数据集的命令
parser.add_argument(
        'dataset',
        type=str,
        choices=[
            'ucf101', 'kinetics400', 'kinetics600', 'kinetics700', 'thumos14',
            'sthv1', 'sthv2', 'mit', 'mmit', 'activitynet', 'hmdb51', 'jester',
            'diving48', 'uavhuman'
        ],
        help='dataset to be built file list')

# 数据集的选择与判断
if args.dataset == 'ucf101':
        splits = parse_ucf101_splits(args.level)
    elif args.dataset == 'sthv1':
        splits = parse_sthv1_splits(args.level)
    elif args.dataset == 'sthv2':
        splits = parse_sthv2_splits(args.level)
    elif args.dataset == 'mit':
        splits = parse_mit_splits()
    elif args.dataset == 'mmit':
        splits = parse_mmit_splits()
    elif args.dataset in ['kinetics400', 'kinetics600', 'kinetics700']:
        splits = parse_kinetics_splits(args.level, args.dataset)
    elif args.dataset == 'hmdb51':
        splits = parse_hmdb51_split(args.level)
    elif args.dataset == 'jester':
        splits = parse_jester_splits(args.level)
    elif args.dataset == 'diving48':
        splits = parse_diving48_splits()
    elif args.dataset == 'uavhuman':
        splits = parse_uavhuman_splits(args.level)
```

最后执行下面语句生成用于训练的文件list
```shell
# 在当前项目目录下(mmaction2/)
PYTHONPATH=. python tools/data/build_file_list.py uavhuman /data/dataset/uavhuman/videos --level 2 --num-split 3 --format videos --shuffle
```

### 3. 配置config并训练

```shell
python tools/train.py work_dir/uavhuman/c3d_pretrained_sports1m_uavhuman_rgb_video.py --work-dir results/checkpoints --seed=0 --deterministic
```


- [ ] 类别序号可能存在问题，**需要注意label从0开始**
- [ ] dataloader错误，**可能数据太大**，抽帧

https://blog.csdn.net/qq_29761909/article/details/126727152


## SlowFast复现

是