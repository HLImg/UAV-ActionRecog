## 视频抽帧并打成nori

### 数据准备
为了尽量减少对mmaction原有代码和逻辑的改动，将遵循以下约束并实现：

1. 使用原有的数据分割文件，不进行任何改动，如[ucf101原始帧的分割文件](../data/ucf101/ucf101_train_split_1_rawframes.txt)所示
    ```
    [cls_name/video_name]~[num_frames]~[label/cls_id] 
    ```
2. 将视频抽帧并打成nori2后，生成一个json格式的索引文件:
    ```
    [cls_name/video_name] : {
        "img_{:05}.jpg": nid，
                ···
                ···
                ···
        "img_{:05}.jpg": nid，
    }
    ``` 

3. 将该json索引文件的oss地址或本地路径当作参数传入到[自定义的RawFrameDecode类中](../mmaction/datasets/transforms/nori_loading.py)，具体来说，在配置文件中
    ```PYTHON
    file_client_args = dict(
        io_backend='oss',
        nids_files = "s3://lianghao02-video/.../train_split_1_rawframes.json
    )
    ```

### 解码 