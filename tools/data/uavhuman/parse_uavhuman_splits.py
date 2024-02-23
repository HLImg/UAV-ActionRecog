# -*- encoding: utf-8 -*-
# @Author  :   Liang Hao 
# @Time    :   2024/02/23 15:19:01
# @File    :   parse_uavhuman_splits.py
# @Contact :   lianghao@whu.edu.cn

import os.path as osp

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
    for i in range(1, 4):
        with open(train_file_template.format(i), 'r') as fin:
            train_list = [line_to_map(x) for x in fin]

        with open(test_file_template.format(i), 'r') as fin:
            test_list = [line_to_map(x) for x in fin]
        splits.append((train_list, test_list))

    return splits


if __name__ == '__main__':
    splits = parse_uavhuman_splits(2)
    print(splits)