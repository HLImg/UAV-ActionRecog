# -*- encoding: utf-8 -*-
# @Author  :   Liang Hao 
# @Time    :   2024/02/23 16:02:55
# @File    :   ann.py
# @Contact :   lianghao@whu.edu.cn

import os.path as osp


def process(root_path, save_dir):
    basename = osp.basename(root_path)
    save_path = osp.join(save_dir, basename)
    
    refs = {}
    with open('annotations/testlist01.txt', 'r') as file:
        for line in file.readlines():
            line = line.strip()
            cls_name, name = line.split('/')
            refs[name] = cls_name
    with open('annotations/trainlist01.txt', 'r') as file:
        for line in file.readlines():
            line = line.strip()
            cls_name, name = line.split(' ')[0].split('/')
            refs[name] = cls_name
    
    splits = []
    
    with open(root_path, 'r') as file:
        for line in file.readlines():
            line = line.strip().split('/')[1]
            cls_name = refs[line]
            cls_ind = int(cls_name[1:])
            
            if 'test' in root_path:
                splits.append(cls_name + "/" + line + '\n')
            else:
                splits.append(cls_name + "/" + line + ' ' + str(cls_ind) + '\n')
    
    
    with open(save_path, 'w') as file:
        file.writelines(splits)
    
    print(f"save splits file on {save_path}")
            

def pro_cls_num(file, save_dir):
    data = []
    basename = osp.basename(file)
    save_path = osp.join(save_dir, basename)
    with open(file, 'r') as file:
        for line in file.readlines():
            line = line.strip().split(' ')
            line[1] = str(int(line[1]) - 1 )
            line = ' '.join(line) + '\n'
            
            data.append(line)
            
    with open(save_path, 'w') as file:
        file.writelines(data)
    
    print(f"save new splits on {save_path}")
    
    
if __name__ == '__main__':
    # root_paths = [
    #                 'annotations/testlist02.txt',
    #                 'annotations/testlist03.txt',
    #                 'annotations/trainlist02.txt',
    #                 'annotations/trainlist03.txt'
    #              ]
    # save_dir = '/data/dataset/uavhuman'
    
    # for root_path in root_paths:
    #     process(root_path, save_dir=save_dir)
    
    paths = [
        'uavhuman_train_split_1_videos.txt',
        'uavhuman_train_split_2_videos.txt',
        'uavhuman_train_split_3_videos.txt',
        'uavhuman_val_split_1_videos.txt',
        'uavhuman_val_split_2_videos.txt',
        'uavhuman_val_split_3_videos.txt'
    ]
    
    save_dir = './new'
    
    for path in paths:
        pro_cls_num(path, save_dir)