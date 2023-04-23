##################################################
# To measure the number of frames for each object
##################################################
import os
import os.path as osp
import numpy as np
import json
import random
import tqdm

path_1 = '/root/autodl-tmp/datasets/refer-yv-2019/Youtube-VOS/train/new1.json'
path_2 = '/root/autodl-tmp/datasets/refer-yv-2019/Youtube-VOS/train/meta_expressions.json'
# data = json.load(open('/root/autodl-tmp/datasets/refer-yv-2019/Youtube-VOS/train/new1.json'))

def test_frames_number(path):
    data = json.load(open(path))
    for vid, objs in tqdm.tqdm(data['videos'].items(), desc='Data processing'):
        # print(len(data['videos'].keys()))
        for obj_id, obj in objs['objects'].items():
            frames_count = len(obj['frames'])
            if frames_count < 3:
                print("less than 3, {},{}".format(frames_count, vid))

def test_sents_number(path):
    data = json.load(open(path))
    total_exp_num = 0
    for vid, objs in tqdm.tqdm(data['videos'].items(), desc='Data processing'):
        exp_num = len(data['videos'][vid]["expressions"].keys())
        total_exp_num += exp_num 
    print(total_exp_num) 
    return total_exp_num
 
if __name__ == "__main__" :
    test_sents_number(path_2)
