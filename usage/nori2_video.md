## 视频抽帧并打成nori

### 数据准备
为了尽量减少对mmaction原有代码和逻辑的改动，将遵循以下约束并实现：

1. 将视频抽帧并打成nori2后，生成一个json格式的索引文件:
    ```
    [cls_name/video_name] : {
        "img_{:05}.jpg": nid，
                ···
                ···
                ···
        "img_{:05}.jpg": nid，
    }
    ``` 
2. 使用原有的数据分割文件，不进行任何改动，如[ucf101原始帧的分割文件](../data/ucf101/ucf101_train_split_1_rawframes.txt)所示
    ```
    [cls_name/video_name]~[num_frames]~[label/cls_id] 
    ```

3. 将nori的json文件和分割文件的oss地址或本地路径当作参数传入到[自定义的RawFrameDecode类中](../mmaction/datasets/transforms/nori_loading.py)，具体来说，在配置文件中
    ```PYTHON
    file_client_args = dict(
        io_backend='oss',
        dtype = 'uint8',
        retry = 60, 
        nori_file = "../data/ucf101/ucf101_train_split_1_nid.json"
    )
    ```

#### UCF101


1. 抽帧，打成nori2文件
```shell
cd tools/data/

python build_rawframe_nori2.py /data/dataset/ucf101/videos/ /data/dataset/ucf101/rawframes/ --task rgb --level 2 --ext avi --new-width 320 --new-height 240 --use-opencv --oss_name lianghao02-video --nori_path dataset/video_action/ucf101_rawframe.nori --json_path /data/workspace/mmaction2/data/ucf101/ucf101.json --num-worker 16
```
2. 根据数据集的训练和测试分割文件，生成训练和验证时的json文件，主要根据抽取帧的名字（动作类名和视频名）来获取每一帧的nid

```shell
cd tools/data/

python build_index_nori2.py --dataset ucf101 --src_folder /data/workspace/mmaction2/data/ucf101 --subset train --num_split 1 --nid_file /data/workspace/mmaction2/data/ucf101/ucf101.json

python build_index_nori2.py --dataset ucf101 --src_folder /data/workspace/mmaction2/data/ucf101 --subset val --num_split 1 --nid_file /data/workspace/mmaction2/data/ucf101/ucf101.json
```

#### UAVHuman
```shell
cd tools/data/
python build_rawframe_nori2.py /data/dataset/uavhuman/videos/ /data/dataset/uavhuman/rawframes/ --task rgb --level 2 --ext avi --use-opencv --oss_name lianghao02-video --nori_path dataset/video_action/uavhuman_rawframe.nori --json_path /data/workspace/mmaction2/data/uavhuman/uavhuman.json --num-worker 24

python build_index_nori2.py --dataset uavhuman --src_folder /data/workspace/mmaction2/data/uavhuman --subset train --num_split 1 --nid_file /data/workspace/mmaction2/data/uavhuman/uavhuman.json

python build_index_nori2.py --dataset uavhuman --src_folder /data/workspace/mmaction2/data/uavhuman --subset val --num_split 1 --nid_file /data/workspace/mmaction2/data/uavhuman/uavhuman.json
```


### 解码 

正常本地抽帧的配置文件如下
```python
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
```

使用oss（nori2）抽帧的配置文件如下

```python
file_client_args = dict(
    io_backend='disk',
    nori_file = 'data/ucf101/ucf101_train_split_1_nid.json',
    dtype = 'uint8',
    retry = 60
)

train_pipeline = [
    # 将视频转成帧，如果直接用帧则注释
    # dict(type='DecordInit', **file_client_args),
    dict(type='SampleFrames', clip_len=16, frame_interval=1, num_clips=1),
    # dict(type='DecordDecode'),
    dict(type='RawFrameDecodeNoir2', **file_client_args),
    dict(type='Resize', scale=(-1, 128)),
    dict(type='RandomCrop', size=112),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]
```

### C3D-UCF101-Nori2

```shell
bash tools/dist_train.sh work_dir/c3d/c3d_pretrained_ncf101_nori.py 1 --work-dir results/c3d_ucf101_nori --seed=0
```
相关的配置文件在[work_dir/c3d](../work_dir/c3d/)目录下，这面展示了本地抽帧和nori抽帧配置文件的差异

```python
--- c3d_pretrained_ncf101_nori.py	2024-03-24 15:46:20.950437836 +0800
+++ c3d_pretrained_ucf101_rgb.py	2024-02-23 10:22:25.396394506 +0800
@@ -21,19 +21,7 @@
 ann_file_test = f'/data/dataset/ucf101/ucf101_val_split_{split}_rawframes.txt'
 ann_file_val = f'/data/dataset/ucf101/ucf101_val_split_{split}_rawframes.txt'
 
-file_client_args_train = dict(
-    io_backend='disk',
-    nori_file = 'data/ucf101/ucf101_train_split_1_nid.json',
-    dtype = 'uint8',
-    retry = 60
-)
-
-file_client_args_eval = dict(
-    io_backend='disk',
-    nori_file = 'data/ucf101/ucf101_val_split_1_nid.json',
-    dtype = 'uint8',
-    retry = 60
-)
+file_client_args = dict(io_backend='disk')
 
 # dataset pipeline
 train_pipeline = [
@@ -41,7 +29,7 @@
     # dict(type='DecordInit', **file_client_args),
     dict(type='SampleFrames', clip_len=16, frame_interval=1, num_clips=1),
     # dict(type='DecordDecode'),
-    dict(type='RawFrameDecodeNoir2', **file_client_args_train),
+    dict(type='RawFrameDecode', **file_client_args),
     dict(type='Resize', scale=(-1, 128)),
     dict(type='RandomCrop', size=112),
     dict(type='Flip', flip_ratio=0.5),
@@ -58,7 +46,7 @@
         num_clips=1,
         test_mode=True),
     # dict(type='DecordDecode'),
-    dict(type='RawFrameDecodeNoir2', **file_client_args_eval),
+    dict(type='RawFrameDecode', **file_client_args),
     dict(type='Resize', scale=(-1, 128)),
     dict(type='CenterCrop', crop_size=112),
     dict(type='FormatShape', input_format='NCTHW'),
@@ -74,7 +62,7 @@
         num_clips=10,
         test_mode=True),
     # dict(type='DecordDecode'),
-    dict(type='RawFrameDecodeNoir2', **file_client_args_eval),
+    dict(type='RawFrameDecode', **file_client_args),
     dict(type='Resize', scale=(-1, 128)),
     dict(type='CenterCrop', crop_size=112),
     dict(type='FormatShape', input_format='NCTHW'),

```