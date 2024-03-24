# -*- coding: utf-8 -*-
# @Time : 2024/03/22 13:09
# @Author : Liang Hao
# @FileName : nori_loading.py
# @Email : lianghao@whu.edu.cn

import cv2
import mmcv
import json
import nori2
import copy as cp
import numpy as np
import os.path as osp


from mmengine.fileio import FileClient
from mmaction.registry import TRANSFORMS
from mmcv.transforms import BaseTransform

nf = nori2.Fetcher()

@TRANSFORMS.register_module()
class RawFrameDecodeNoir2(BaseTransform):
    def __init__(self,
                 nori_file,
                 dtype = 'uint8',
                 retry = 60, 
                 io_backend = 'oss',
                 decoding_backend = 'cv2',
                 **kwargs
                 ):
        super().__init__()
        
        self.retry = retry
        self.dtype = np.dtype(dtype)
        self.io_backend = io_backend
        self.decoding_backend = decoding_backend
        self.kwargs = kwargs
        self.file_client = None
        
        with open(nori_file, 'r') as file:
            self.nids = json.load(file)
    
    def transform(self, results):
        mmcv.use_backend(self.decoding_backend)
        
        directory = results['frame_dir']
        filename_tmpl = results['filename_tmpl']
        modality = results['modality']
        
        # act_name and video_name
        
        act_name, video_name = directory.split('/')[-2:]
        
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend, **self.kwargs)
            
        imgs = list()
        
        # 被采样帧的id
        if results['frame_inds'].ndim != 1:
            results['frame_inds'] = np.squeeze(results['frame_inds'])
        
        offset = results.get('offset', 0)
        
        cache = {}
        
        for i, frame_idx in enumerate(results['frame_inds']):
            if frame_idx in cache:
                imgs.append(cp.deepcopy(imgs[cache[frame_idx]]))
                continue
            else:
                cache[frame_idx] = i
            
            if modality == 'RGB':
                frame_name = filename_tmpl.format(frame_idx)
                nid = self.nids[act_name][video_name][frame_name]
                
                buffer = nf.get(nid, retry=self.retry)
                
                if buffer is None:
                    raise FileExistsError(f"read {nid} error, buffer is None")
                
                im_arr = np.frombuffer(buffer, dtype=self.dtype)
                cur_frame  = cv2.imdecode(im_arr, cv2.IMREAD_UNCHANGED)
                imgs.append(cur_frame)
            else:
                raise NotImplementedError
        
        results['imgs'] = imgs
        results['original_shape'] = imgs[0].shape[:2]
        results['img_shape'] = imgs[0].shape[:2]
        
        # we resize the gt_bboxes and proposals to their real scale
        if 'gt_bboxes' in results:
            h, w = results['img_shape']
            scale_factor = np.array([w, h, w, h])
            gt_bboxes = results['gt_bboxes']
            gt_bboxes = (gt_bboxes * scale_factor).astype(np.float32)
            results['gt_bboxes'] = gt_bboxes
            if 'proposals' in results and results['proposals'] is not None:
                proposals = results['proposals']
                proposals = (proposals * scale_factor).astype(np.float32)
                results['proposals'] = proposals

        return results