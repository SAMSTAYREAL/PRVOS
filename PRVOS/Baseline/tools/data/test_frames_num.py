##################################################
# To measure the number of frames for each object
##################################################
import os
import os.path as osp
import numpy as np
import json
import random
import tqdm

data = json.load(open('/home/imi005/datasets2/datasets/refer-yv-2019/Youtube-VOS/train/new1.json'))

for vid, objs in tqdm.tqdm(data['videos'].items(), desc='Data processing'):
    for obj_id, obj in objs['objects'].items():
        frames_count = len(obj['frames'])
        if frames_count < 3:
            print("less than 3, {},{}".format(frames_count, vid))
