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
        nids_files = "s3://lianghao02-video/.../train_split_1_rawframes.json
    )
    ```

#### UCF101

```shell

cd tools/data/

python build_rawframe_nori2.py /data/dataset/ucf101/videos/ /data/dataset/ucf101/rawframes/ --task rgb --level 2 --ext avi --new-width 320 --new-height 240 --use-opencv --oss_name lianghao02-video --nori_path dataset/video_action/ucf101_rawframe.nori --json_path /data/workspace/mmaction2/data/ucf101/ucf101.json --num-worker 16
```

#### UAVHuman
```shell
cd tools/data/
python build_rawframe_nori2.py /data/dataset/uavhuman/videos/ /data/dataset/uavhuman/rawframes/ --task rgb --level 2 --ext avi --use-opencv --oss_name lianghao02-video --nori_path dataset/video_action/uavhuman_rawframe.nori --json_path /data/workspace/mmaction2/data/uavhuman/uavhuman.json --num-worker 24
```


### 解码 