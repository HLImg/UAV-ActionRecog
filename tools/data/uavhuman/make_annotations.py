# -*- encoding: utf-8 -*-
# @Author  :   Liang Hao 
# @Time    :   2024/02/23 10:39:58
# @File    :   make_annotations.py
# @Contact :   lianghao@whu.edu.cn

import os
import glob

'''
uavhuman/
        train/
            A001
            A002
        test/
            A001
            A002
'''


def classInd(root_dir, save_dir):
    save_path = os.path.join(save_dir, 'classInd.txt')
    
    datas = []
    for name in os.listdir(root_dir):
        cls_ind = int(name[1:])
        datas.append(f"{cls_ind} {name}\n")
    
    with open(save_path, 'w') as file:
        file.writelines(datas)
    
    print(f"classInd.txt is saved on {save_path}.")
    

def train_split(root_dir, save_dir, idx):
    save_name = "trainlist" + str(idx).zfill(2) + ".txt"
    save_path = os.path.join(save_dir, save_name)
    
    datas = []
    for dirname in os.listdir(root_dir):
        sub_dir = os.path.join(root_dir, dirname)
        cls_ind = int(dirname[1:])
        for avi in glob.glob(os.path.join(sub_dir, '*.avi')):
            avi = avi.split('/')
            path = '/'.join(avi[5:])
            datas.append(f"{path} {cls_ind}\n")
    
    with open(save_path, 'w') as file:
        file.writelines(datas)
    
    print(f"{save_name} is saved on {save_path}. The number of videos is {len(datas)}")
    

def test_split(root_dir, save_dir, idx):
    save_name = "testlist" + str(idx).zfill(2) + ".txt"
    save_path = os.path.join(save_dir, save_name)
    
    datas = []
    for dirname in os.listdir(root_dir):
        sub_dir = os.path.join(root_dir, dirname)
        cls_ind = int(dirname[1:])
        for avi in glob.glob(os.path.join(sub_dir, '*.avi')):
            avi = avi.split('/')
            path = '/'.join(avi[5:])
            datas.append(f"{path}\n")
    
    with open(save_path, 'w') as file:
        file.writelines(datas)
    
    print(f"{save_name} is saved on {save_path}. The number of videos is {len(datas)}")
    

if __name__ == '__main__':
    root_dir_train = '/data/dataset/uavhuman/train'
    root_dir_test = '/data/dataset/uavhuman/test'
    save_dir = '/data/dataset/uavhuman/annotations'
    
    # classInd(root_dir_train, save_dir)
    
    # train_split(root_dir_train, save_dir, idx=1)
    # test_split(root_dir_test, save_dir, idx=1)
    
    datas = []
    root_dir = '/data/dataset/uavhuman/rawframes'
    for dirname in os.listdir(root_dir):
        sub_dir = os.path.join(root_dir, dirname)
        for dir_ in os.listdir(sub_dir):
            datas.append(dir_)
    
    print(len(datas))