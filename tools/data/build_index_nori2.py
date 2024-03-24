# -*- coding: utf-8 -*-
# @Time : 2024/03/24 13:43
# @Author : Liang Hao
# @FileName : build_index_nori2.py
# @Email : lianghao@whu.edu.cn

import json
import argparse
import os.path as osp

def parse_args():
    parser = argparse.ArgumentParser(description="Build Index of nori2 files")
    
    parser.add_argument(
        '--dataset', 
        type=str, 
        choices=[
            'ucf101', 'uavhuman'
        ],
        help="the split files for training and testing"
    )
    
    parser.add_argument(
        '--src_folder', type=str, help='root directory for the frames')
    
    parser.add_argument(
        '--nid_file',
        type=str,
        help='the split json files for training and testing'
    )
    
    parser.add_argument(
        '--num_split',
        type=int,
        default=3,
        help='number of split to file list'
    )
    
    parser.add_argument(
        '--subset',
        type=str,
        default='train',
        choices=['train', 'val', 'test'],
        help='subset to generate file list'
    )
    
    args = parser.parse_args()
    
    return args


def build_nid_files(args):
    with open(args.nid_file, 'r') as file:
        total_nids = json.load(file)
        file.close()
    
    flag = False
    split_file = f"{args.dataset}_{args.subset}_split_{args.num_split}_rawframes.txt"
    split_path = osp.join(args.src_folder, split_file)
    
    if not osp.exists(split_path):
        flag = True
        split_file = f"{args.dataset}_{args.subset}_split_{args.num_split}_videos.txt"
        split_path = osp.join(args.src_folder, split_file)
    
    if not osp.exists(split_path):
        raise FileExistsError(f"{split_path}")
    
    save_path = osp.join(args.src_folder, f"{args.dataset}_{args.subset}_split_{args.num_split}_nid.json")
    
    data = dict()
    with open(split_path, 'r') as file:
        for line in file.readlines():
            line = line.strip()
            
            if flag:
                act_name, video_name = line.split(' ')[0].split('/')
                video_name = video_name.split('.')[0]
            else:
                act_name, video_name = line.split(' ')[0].split('/')
            
            if act_name not in data:
                data[act_name] = {}
            
            if video_name not in data[act_name]:
                data[act_name][video_name] = {} 
            
            data[act_name][video_name].update(total_nids[act_name][video_name])    
    
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    
            
if __name__ == '__main__':
    args = parse_args()
    build_nid_files(args)