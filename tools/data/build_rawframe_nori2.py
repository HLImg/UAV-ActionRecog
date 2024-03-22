# -*- coding: utf-8 -*-
# @Time : 2024/03/22 13:26
# @Author : Liang Hao
# @FileName : build_rawframe_nori2.py
# @Email : lianghao@whu.edu.cn

# Copyright (c) OpenMMLab. All rights reserved.

import os
import sys
import glob
import cv2
import os.path as osp

import nori2
import refile
import mmcv
import numpy as np

import warnings
import argparse
import json

from multiprocessing import Lock, Pool


def extract_frame(vid_item):
    full_path, vid_path, vid_id, method, task, report_file = vid_item
    
    if '/' in vid_path:
        act_name = osp.basename(osp.dirname(vid_path))
        out_full_path = osp.join(args.out_dir, act_name)
    else:
        pass
        out_full_path = args.out_dir
        
    data = {}
    data[act_name] = {}
        
    run_success = -1
    
    video_file = dict()
    
    res = []
    
    try:
        video_name = osp.splitext(osp.basename(vid_path))[0]
        vr = mmcv.VideoReader(full_path)   
        
        data_list = []
        filename_list = []
        
        for i, vr_frame in enumerate(vr):
            if vr_frame is not None:
                w, h, _ = np.shape(vr_frame)
                        
                if args.new_short == 0:
                    if args.new_width == 0 or args.new_height == 0:
                        out_img = vr_frame
                    else:
                        out_img = mmcv.imresize(vr_frame,
                                    (args.new_width, args.new_height))
                else:
                    if min(h, w) == h:
                        new_h = args.new_short
                        new_w = int((new_h / h) * w)
                    else:
                        new_w = args.new_short
                        new_h = int((new_w / w) * h)
                    out_img = mmcv.imresize(vr_frame, (new_h, new_w))
                
                # 编码
                _, encoded = cv2.imencode('.png', out_img, [cv2.IMWRITE_PNG_COMPRESSION, 1])
                key = f'img_{i + 1:05d}.jpg'
                
                data_list.append(encoded.tobytes())
                filename_list.append(key)
                
                nid = args.nori_writer.put(encoded.tobytes())
                if not nid:
                    print("None")
                    return 
                video_file[key] = nid
            
            else:
                warnings.warn(
                        'Length inconsistent!'
                        f'Early stop with {i + 1} out of {len(vr)} frames.'
                    )
                break
        
        run_success = 0
            
    except Exception:
        raise f"Error on Ex"
        run_success = -1
        return    
    
    
    data[act_name][video_name] = video_file
    
    # print(data.keys(), data[act_name].keys(), data[act_name][video_name].keys())
    
    return data


def parse_args():
    parser = argparse.ArgumentParser(description='extract optical flows')
    parser.add_argument('src_dir', type=str, help='source video directory')
    parser.add_argument('--oss_name', type=str, help='oss username')
    parser.add_argument('--nori_path', type=str, help='nori path')
    parser.add_argument('--json_path', type=str, help='nori path')
    parser.add_argument('out_dir', type=str, help='output rawframe directory')
    parser.add_argument(
        '--task',
        type=str,
        default='flow',
        choices=['rgb', 'flow', 'both'],
        help='which type of frames to be extracted')
    parser.add_argument(
        '--level',
        type=int,
        choices=[1, 2],
        default=2,
        help='directory level of data')
    parser.add_argument(
        '--num-worker',
        type=int,
        default=8,
        help='number of workers to build rawframes')

    parser.add_argument(
        '--ext',
        type=str,
        default='avi',
        choices=['avi', 'mp4', 'webm'],
        help='video file extensions')
    parser.add_argument(
        '--mixed-ext',
        action='store_true',
        help='process video files with mixed extensions')
    parser.add_argument(
        '--new-width', type=int, default=0, help='resize image width')
    parser.add_argument(
        '--new-height', type=int, default=0, help='resize image height')
    parser.add_argument(
        '--new-short',
        type=int,
        default=0,
        help='resize image short side length keeping ratio')
    parser.add_argument('--num-gpu', type=int, default=8, help='number of GPU')
    parser.add_argument(
        '--resume',
        action='store_true',
        default=False,
        help='resume optical flow extraction instead of overwriting')
    parser.add_argument(
        '--use-opencv',
        action='store_true',
        help='Whether to use opencv to extract rgb frames')
    parser.add_argument(
        '--input-frames',
        action='store_true',
        help='Whether to extract flow frames based on rgb frames')
    parser.add_argument(
        '--flow-type',
        type=str,
        default=None,
        choices=[None, 'tvl1', 'warp_tvl1', 'farn', 'brox'],
        help='flow type to be generated')
    parser.add_argument(
        '--report-file',
        type=str,
        default='build_report.txt',
        help='report to record files which have been successfully processed')
    args = parser.parse_args()

    return args

def init(lock_):
    global lock
    lock = lock_
    
def merge_dicts(dicts):
    result = {}
    for d in dicts:
        for act_name, videos in d.items():
            if act_name not in result:
                result[act_name] = {}
            for video_name, frames in videos.items():
                if video_name not in result[act_name]:
                    result[act_name][video_name] = {}
                result[act_name][video_name].update(frames)
    return result


if __name__ == '__main__':
    args = parse_args()
    
    if args.input_frames:
        print('Reading rgb frames from folder: ', args.src_dir)
        fullpath_list = glob.glob(args.src_dir + '/*' * args.level)
        print('Total number of rgb frame folders found: ', len(fullpath_list))
    
    else:
        print('Reading videos from folder: ', args.src_dir)
        if args.mixed_ext:
            print('Extension of videos is mixed')
            fullpath_list = glob.glob(args.src_dir + '/*' * args.level)
        else:
            print('Extension of videos: ', args.ext)
            fullpath_list = glob.glob(args.src_dir + '/*' * args.level + '.' +
                                      args.ext)
        print('Total number of videos found: ', len(fullpath_list))
    
    
    if args.resume:
        done_fullpath_list = []
        with open(args.report_file) as f:
            for line in f:
                if line == '\n':
                    continue
                done_full_path = line.strip().split()[0]
                done_fullpath_list.append(done_full_path)
        done_fullpath_list = set(done_fullpath_list)
        fullpath_list = list(set(fullpath_list).difference(done_fullpath_list))
    
    if args.level == 2:
        vid_list = list(
            map(
                lambda p: osp.join(
                    osp.basename(osp.dirname(p)), osp.basename(p)),
                fullpath_list))
    elif args.level == 1:
        vid_list = list(map(osp.basename, fullpath_list))
        
    lock = Lock()
    
    args.oss_name = "s3://" + args.oss_name
    args.nori_path = os.path.join(args.oss_name, args.nori_path)
    
    # 初始化nori2对象
    
    nori_writer = nori2.remotewriteopen(args.nori_path)
    
    args.nori_writer = nori_writer
    
    pool = Pool(args.num_worker, initializer=init, initargs=(lock, ))
    result_dicts = pool.map(
        extract_frame,
        zip(fullpath_list, vid_list, range(len(vid_list)),
            len(vid_list) * [args.flow_type],
            len(vid_list) * [args.task],
            len(vid_list) * [args.report_file]))
    pool.close()
    pool.join()
    
    args.nori_writer.join()
    
    merged_res = merge_dicts(result_dicts)
    
    with open(args.json_path, 'w', encoding='utf-8') as f:
        json.dump(merged_res, f, ensure_ascii=False, indent=4)
        
    
    print(f"[>>>>>>>>>>>>>>>>] nori speedup {args.nori_path} --on --replica=2")