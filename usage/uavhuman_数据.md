
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
