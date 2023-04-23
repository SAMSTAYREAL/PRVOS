import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.autograd import Variable
import numpy as np
from PIL import Image
from models.model import Encoder_Q, Encoder_M, Memorize
from torch.utils import data
from pathlib import Path
import json
from tqdm import tqdm
import random
import os

frame_feature_dir = './frame_features'
mask_feature_dir = './mask_features'
data_list = []
data_root = '/home/imi005/datasets2/datasets/STM/data/Youtube-VOS'

class REFER_YV_2019_EXTRACTOR(data.Dataset):
    def __init__(self, data_root, split, N = 2, scale = 1.0):
        self.data_root = Path(data_root)
        split_type = split.split('_')[0]
        self.split = split_type
        self.N = N
        self.scale = scale
        self.max_frames = 36
        self.image_dir = self.data_root / split / 'JPEGImages' # split 为train or eval or test
        self.mask_dir = self.data_root / split / 'Annotations'
    
    def get_info(self):
        data = json.load(open(self.data_root / self.split / 'new1.json'))
        
        self.videos = []
        for vid, objs in tqdm.tqdm(data['videos'].items(), desc = 'Data processing'):
            for obj_id, obj in objs['objects'].items():
                oid = int(obj_id)
                for frm in obj['frames']:
                    mask_name = self.data_root / self.split / 'Annotations' / vid / '{}.png'.format(frm)
                    mask = np.uint8(Image.open(mask_name).convert('P'))
                    mask = np.uint8(mask == oid)
                self.videos.append([vid, oid, obj['category'], obj['frames']])
        
        len_videos = len(self.videos)
        if self.scale < 1.0:
            len_videos = int(len_videos * self.scale)
        self.videos = self.videos[:len_videos]
                      
                
    
    def __len__(self):
        len_videos = len(self.videos)
        return len_videos
    
    def resize(self, frame, mask, size):
        scale = np.maximum(size[0]/np.float(frame.shape[0]), size[1]/np.float(frame.shape[1]))
        dsize = (np.int(frame.shape[1]*scale), np.int(frame.shape[0]*scale))
        size = (size[0], size[1])
        resize_frame  = cv2.resize(frame, dsize=size, interpolation=cv2.INTER_LINEAR)
        resize_mask = cv2.resize(mask, dsize=size, interpolation=cv2.INTER_NEAREST)
        return resize_frame, resize_mask  
    
    # 加载一对原图和对应的mask
    def load_pair(self, vid, oid, fid):
        img_name = self.data_root / self.split / 'JPEGImages' / vid / '{}.jpg'.format(fid)
        mask_name = self.data_root / self.split / 'Annotations' / vid / '{}.png'.format(fid)
    
        frame = np.float32(Image.open(img_name).convert('RGB')) / 255.
        mask = np.uint8(Image.open(mask_name).convert('P'))
        mask = np.uint8(mask == oid)
        return frame, mask  
    
    # 加载需要的原图和mask
    def load_pairs(self, vid, oid, frame_ids):
        frames, masks = [], []
        for frame_id in frame_ids:
            frame, mask = self.load_pair(vid, oid, frame_id)
            frame, mask = self.resize(frame, mask, self.size)
            frames.append(frame)
            masks.append(mask)
            
        N_frames = np.stack(frames, axis=0) # 将每个frame和mask堆叠
        N_masks = np.stack(masks, axis=0)
        
        Fs = torch.from_numpy(np.transpose(N_frames, (0, 3, 1, 2)).copy()).float()  # origin: (0,3,1,2)
        Ms = torch.from_numpy(N_masks.copy()).float()
        return Fs, Ms
    
    def __getitem__(self, index):
        
        rnd = random.Random()
        vid, oid, category, frames_ids = self.videos[index]
        frames, masks = [], []
        for i in len(frames_ids):
            frm, msk = self.load_pair(vid, oid, frames_ids[i])
            if self.jitter:
                frm, msk = self.random_jitter(frm, msk, self.size, rnd)
            else:
                frm, msk = self.resize(frm, msk, self.size)
            frames.append(frm)
            masks.append(msk)
        
        frames = np.stack(frames, axis=0)
        masks = np.stack(masks, axis=0)
        
        Fs = torch.from_numpy(np.transpose(frames, (0, 3, 1, 2)).copy()).float()    # 转换成tensor， （num_frames, C, H, W）
        Ms = torch.from_numpy(masks.copy()).float()
        
        return Fs, Ms
        
def get_dataset(dataset, DATA_ROOT, N, batch_size, img_size=(320, 320), max_skip=2, jitter=True, scale=1.0):
    
    if 'refer-yv-2019' in dataset:
        split_type = 'train_full'
        
        # trainset = REFER_YV_2019(data_root=DATA_ROOT / 'youtube-vos-2019', split=split_type, N=N, size=img_size, max_skip=max_skip, jitter=jitter, scale=scale)
        trainset = REFER_YV_2019_EXTRACTOR(data_root=os.path.join(DATA_ROOT,'Youtube-VOS'), split=split_type, N=N, size=img_size, max_skip=max_skip, jitter=jitter, scale=scale)
        dataLoader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True, pin_memory=True)
    else:
        raise ValueError
        
    return trainset, dataLoader

class feature_extractor():
    def __init__(self, backbone = 'resnet50'):
        super(feature_extractor, self).__init__()
        self.Encoder_Q = Encoder_Q
        self.Encoder_M = Encoder_M
        self.Memorize = Memorize
        
    def forward(self, input_f_q, input_f_m, input_m):
        
        
        
        
    
    