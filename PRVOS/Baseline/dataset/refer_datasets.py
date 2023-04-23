import sys
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
# sys.path.remove('/home/imi005/students/CZQ/Refer-Youtube-VOS/Baseline/utils')
import os
import os.path as osp
import numpy as np
from PIL import Image

import collections
import torch
import torchvision
from torch.utils import data

import random
import scipy.io
import glob, pdb
import time
import cv2  # ros: sudo mv cv2.so cv2_ros.so
import random
import csv
import json
import pickle
from tqdm import tqdm
import torch.nn.functional as F

from itertools import chain, combinations

import tqdm

import matplotlib.pyplot as plt

def _flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x

#####
from transformers import AutoTokenizer, DistilBertModel, BertConfig, BertTokenizer
#####

import pdb
from torch.utils import data
from pathlib import Path
from time import time
from utils.word_utils import Corpus
from dataset.aug import *

from utils.pseudo_label_utils import *
import opts

import collections


NEW_JSON = '/home/imi1214/MJP/datasets/RVOS/refer-yv-2019/Youtube-VOS/valid/new1.json'
META_EXPRESSION_JSON = '/home/imi1214/MJP/datasets/RVOS/refer-yv-2019/Youtube-VOS/meta_expressions_test/meta_expressions/valid/meta_expressions.json'

##### get corresponding expression id #####
##### use in set_meta_file() #####
def get_exp_id(vid, sent):
    # print(vid)
    # with open(NEW_JSON, 'r') as f1:
    #     json_new = json.load(f1)
    with open(META_EXPRESSION_JSON, 'r') as f2:
        json_meta_expressions = json.load(f2)
    
    exp_id_new = None
    exp_meta = json_meta_expressions["videos"][vid]["expressions"]
    frame_ids = list(json_meta_expressions["videos"][vid]["frames"])
    # exp_new = json_new["videos"][vid]["objects"][obj_id]
    for exp_id in exp_meta.keys():
        # print(exp_id)
        exp_ = exp_meta[exp_id]["exp"]
        if exp_ == sent:
            exp_id_new = exp_id
        else:
            continue
        # print(exp_)
        # for exp in list(json_new["videos"][vid]["objects"][obj_id]["expressions"]):
        #     if exp_ == exp:
        #         exp_id = exp_id
        #         break       
    return exp_id_new, frame_ids

def get_epoch_number():
    args = opts.get_args_parser()
    epoch = args.epoch
    return epoch
        
class REFER_YV_2019(data.Dataset):
    
    def __init__(self, data_root, split, N=2, size=(256,256), max_skip=1, query_len=20, mode='', jitter=True, bert=False, scale=1.0):
        self.data_root = Path(data_root)
        # self.data_root = '/home/imi005/datasets2/datasets/STM/data/Youtube-VOS'
       
        split_type = split.split('_')[0]
    
        self.split = split_type
        
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.text_model = DistilBertModel.from_pretrained("distilbert-base-uncased")
        
        self.N = N
        self.size = size
        self.query_len = query_len
        self.mode = mode
        self.jitter = jitter
        self.bert = bert
        self.scale = scale
        
        self.max_frames = 48
        self.skip = 1   # use for selecting frames
        
        self.image_dir = self.data_root / split / 'JPEGImages' 
        self.mask_dir = self.data_root / split / 'Annotations'
        
        self.set_meta_file()
    
        
    def set_meta_file(self):

        mymeta_path = self.data_root / self.split / 'mymeta.pkl'
        if mymeta_path.exists():
            with mymeta_path.open('rb') as f:
                self.videos = pickle.load(f)
        else:
            data = json.load(open(self.data_root / self.split / 'new1.json'))
            
            self.videos = []
            for vid, objs in tqdm.tqdm(data['videos'].items(), desc='Data processing'):
                #################### evaluation on refer-yv-2019-valid ####################
                if self.mode == 'eval_yv':
                    for obj_id, obj in objs['objects'].items():
                        oid = int(obj_id)
                        frames_count = len(obj['frames'])
                        num_exp = len(obj["expressions"])
                        if frames_count < 3:
                            print(vid)
                            print('less than 3')
                        
                        for i in range(num_exp): 
                            sent = list(obj['expressions'])[i]
                            exp_id, frame_ids = get_exp_id(vid, sent)
                            if exp_id is None:
                                print('wrong file is {}_{}'.format(vid, i))
                            self.videos.append([vid, oid, obj['category'], frame_ids, sent, exp_id, frames_count])
                            
                #################### evaluation on refer-davis-2017-valid ####################            
                elif self.mode == 'eval_davis':
                    object_list = []
                    for obj_id in objs['objects']:
                        object_list.append(obj_id)
                        continue
                        
                    for obj_id, obj in objs['objects'].items():
                        oid = int(obj_id)
                        num_objects = len(object_list)
                        frames_count = len(obj['frames'])
                        if frames_count < 3:
                            print('less than 3')
                        
                        sents = obj['expressions']
                        if len(sents) > 0:
                            self.videos.append([vid, oid, obj['category'], obj['frames'], sents[0], num_objects])   # 添加exp_id
                
                #################### forward only on refer-yv-2019-train ####################
                elif self.mode == 'eval_yv_forward':
                    object_list = []
                    for obj_id in objs['objects']:
                        object_list.append(obj_id)
                        continue
                        
                    for obj_id, obj in objs['objects'].items():
                        oid = int(obj_id)
                        num_objects = len(object_list)
                        frames_count = len(obj['frames'])
                        num_exp = len(obj["expressions"])
                        if frames_count < 3:
                            print('less than 3')
                        for i in range(num_exp):
                            sent = list(obj['expressions'])[i]
                            exp_id = get_exp_id(vid, sent) 
                            self.videos.append([vid, oid, obj['frames'], sent, exp_id, frames_count]) 
                
                #################### training on refer-yv-2019-train ####################
                elif self.mode == 'train_yv':
                    object_list = []
                    for obj_id in objs['objects']:
                        object_list.append(obj_id)
                        continue
                        
                    for obj_id, obj in objs['objects'].items():
                        oid = int(obj_id)
                        num_objects = len(object_list)
                        frames_count = len(obj['frames'])
                        num_exp = len(obj["expressions"])    
                        
                        for i in range(num_exp):
                            sent = list(obj['expressions'])[i]
                            exp_id = get_exp_id(vid, sent) 
                            self.videos.append([vid, oid, obj['category'], obj['frames'], sent, exp_id])

                #################### training on ref-davis-train ####################        
                elif self.mode == 'train_davis':
                    object_list = []
                    for obj_id in objs['objects']:
                        object_list.append(obj_id)
                        continue
                        
                    for obj_id, obj in objs['objects'].items():
                        oid = int(obj_id)
                        num_objects = len(object_list)
                        frames_count = len(obj['frames'])
                        if frames_count < 3:
                            print('less than 3')
                        
                        sents = obj['expressions']
                        if len(sents) > 0:
                            self.videos.append([vid, oid, obj['category'], obj['frames'], sents[0], num_objects])
                else:
                    raise ValueError
                
            
            with mymeta_path.open('wb') as f:
                pickle.dump(self.videos, f, pickle.HIGHEST_PROTOCOL)
                
        len_videos = len(self.videos)
        if self.scale < 1.0:
            len_videos = int(len_videos * self.scale)
        self.videos = self.videos[:len_videos]


    def __len__(self):
        len_videos = len(self.videos)
        return len_videos
    
    
    ##### set corpus #####   
    ##### Unused ######### 
    def set_corpus(self):
        self.corpus = Corpus()
        #TODO: ref file
        # vocab_path = self.data_root / 'vocabulary_Gref.txt'
        # corpus_path = self.data_root / 'corpus.pth'
        vocab_path = os.path.join(self.data_root,'vocabulary_Gref.txt')
        corpus_path = os.path.join(self.data_root,'corpus.pth')
        # if not corpus_path.exists():
        if not os.path.exists(corpus_path):
            print('Saving dataset corpus dictionary...')
            self.corpus.load_file(vocab_path)
            torch.save(self.corpus, corpus_path)
        else:
            self.corpus = torch.load(corpus_path)

    
    ##### random crop #####       
    def random_crop(self, frame, mask, size, rnd):

        # resize `frame` before cropping
        # resized frame should be large than `size` but shouldn't be too large
        min_scale = np.maximum(size[0]/np.float(frame.shape[0]), size[1]/np.float(frame.shape[1]))
        scale = np.maximum(rnd.uniform(min_scale+0.01, 1.875*min_scale), min_scale+0.01)

        dsize = (np.int(frame.shape[1]*scale), np.int(frame.shape[0]*scale))
        trans_frame  = cv2.resize(frame, dsize=dsize, interpolation=cv2.INTER_LINEAR)
        trans_mask = cv2.resize(mask, dsize=dsize, interpolation=cv2.INTER_NEAREST)
        
        ## try to crop patch that contains object area if possible, otherwise just return
        np_in1 = np.sum(trans_mask)

        for _ in range(100):
            cr_y = rnd.randint(0, trans_mask.shape[0] - size[0])
            cr_x = rnd.randint(0, trans_mask.shape[1] - size[1])
            crop_mask = trans_mask[cr_y:cr_y+size[0], cr_x:cr_x+size[1]]
            crop_frame = trans_frame[cr_y:cr_y+size[0], cr_x:cr_x+size[1],:]

            nnz_crop_mask = np.sum(crop_mask)
            break

        return crop_frame, crop_mask
    
    
    ##### random jitter #####
    def random_jitter(self, frame, mask, size, rnd):

        scale = rnd.uniform(1, 1.1)
        dsize = (int(size[0]*scale), int(size[1]*scale))

        trans_frame  = cv2.resize(frame, dsize=dsize, interpolation=cv2.INTER_LINEAR)
        trans_mask = cv2.resize(mask, dsize=dsize, interpolation=cv2.INTER_NEAREST)
        
        np_in1 = np.sum(trans_mask)

        crop_frame = None
        for _ in range(100):
            cr_y = rnd.randint(0, trans_mask.shape[0] - size[0])
            cr_x = rnd.randint(0, trans_mask.shape[1] - size[1])
            crop_mask = trans_mask[cr_y:cr_y+size[0], cr_x:cr_x+size[1]]
            crop_frame = trans_frame[cr_y:cr_y+size[0], cr_x:cr_x+size[1],:]
            if np.sum(crop_mask) > 0.8*np_in1:
                break
                
        if crop_frame is None:
            return self.random_jitter(frame, mask, size, rnd)

        return crop_frame, crop_mask


    ##### resize frames and masks #####
    ##### use for training #####
    def resize(self, frame, mask, size):
        scale = np.maximum(size[0]/np.float(frame.shape[0]), size[1]/np.float(frame.shape[1]))
        dsize = (np.int(frame.shape[1]*scale), np.int(frame.shape[0]*scale))
        size = (size[0], size[1])
        resize_frame  = cv2.resize(frame, dsize=size, interpolation=cv2.INTER_LINEAR)
        resize_mask = cv2.resize(mask, dsize=size, interpolation=cv2.INTER_NEAREST)
        return resize_frame, resize_mask
    
    
    ##### resize frame #####
    ##### use for evaluation #####
    def resize_frame(self, frame, size):
        scale = np.maximum(size[0]/np.float(frame.shape[0]), size[1]/np.float(frame.shape[1]))
        dsize = (np.int(frame.shape[1]*scale), np.int(frame.shape[0]*scale))
        size = (size[0], size[1])
        resize_frame  = cv2.resize(frame, dsize=size, interpolation=cv2.INTER_LINEAR)
        return resize_frame
    
    
    ##### resize mask #####
    ##### use for evaluation #####
    def resize_mask(self, mask, size):
        size = (size[0], size[1])
        resize_mask = cv2.resize(mask, dsize=size, interpolation=cv2.INTER_NEAREST)
        return resize_mask
    
    
    ##### Load a pair of frame and mask #####
    def load_pair(self, vid, oid, fid):
        img_name = self.data_root / self.split / 'JPEGImages' / vid / '{}.jpg'.format(fid)
        mask_name = self.data_root / self.split / 'Annotations' / vid / '{}.png'.format(fid)
    
        frame = np.float32(Image.open(img_name).convert('RGB')) / 255.
        mask = np.uint8(Image.open(mask_name).convert('P'))
        # print("MASK SHAPE IS {}".format(mask.shape))
        mask = np.uint8(mask == oid)
        return frame, mask
    
    
    ##### Load needed frame & mask #####
    def load_pairs(self, vid, oid, frame_ids):
        frames, masks = [], []
        for frame_id in frame_ids:
            frame, mask = self.load_pair(vid, oid, frame_id)
            frame, mask = self.resize(frame, mask, self.size)
            frames.append(frame)
            masks.append(mask)
            
        N_frames = np.stack(frames, axis=0)
        N_masks = np.stack(masks, axis=0)
        
        Fs = torch.from_numpy(np.transpose(N_frames, (0, 3, 1, 2)).copy()).float()  # origin: (0,3,1,2)
        Ms = torch.from_numpy(N_masks.copy()).float()
        return Fs, Ms
    
    
    ##### Load data for valid set #####
    def load_valid(self, vid, oid, frame_ids):
        # print(vid)
        frames, masks = [], []
        mask_path = self.data_root / self.split / 'Annotations' / vid
        mask_file_list = os.listdir(mask_path)
        # print(mask_file_list)
        mask_name = self.data_root / self.split / 'Annotations' / vid / '{}'.format(mask_file_list[0])
        mask = Image.open(mask_name).convert('P')
        initial_size = list(mask.size)
        print(initial_size)
        mask = np.uint8(mask)
        mask = np.uint8(mask == oid)
        
        mask = self.resize_mask(mask, self.size)
        masks.append(mask)

        for frame_id in frame_ids:
            img_name = self.data_root / self.split / 'JPEGImages' / vid / '{}.jpg'.format(frame_id)
            frame = np.float32(Image.open(img_name).convert('RGB')) / 255.
            frame= self.resize_frame(frame, self.size)
            frames.append(frame)
        
        N_frames = np.stack(frames, axis=0) 
        N_masks = np.stack(masks, axis=0)
        
        Fs = torch.from_numpy(np.transpose(N_frames, (0, 3, 1, 2)).copy()).float()  # origin: (0,3,1,2)
        Ms = torch.from_numpy(N_masks.copy()).float()  
        return Fs, Ms, initial_size
    
        
    ##### Get needed frame ids ####
    def sample_frame_ids_base(self, frame_ids):
        mem_frame_ids = []
        frames_count = len(frame_ids)
        if frames_count >= 5:
            n1 = random.sample(range(0,frames_count-4), 1)[0]
            n2 = random.sample(range(n1+1, min(frames_count-3,n1+4+self.skip)),1)[0]
            n3 = random.sample(range(n2+1, min(frames_count-2,n2+3+self.skip)),1)[0]
            n4 = random.sample(range(n2+1, min(frames_count-1,n2+2+self.skip)),1)[0]
            n5 = random.sample(range(n2+1, min(frames_count,n2+1+self.skip)),1)[0]
            frame_1 = frame_ids[n1]
            frame_2 = frame_ids[n2]
            frame_3 = frame_ids[n3]
            frame_4 = frame_ids[n4]
            frame_5 = frame_ids[n5]
            mem_frame_ids = [frame_1, frame_2, frame_3, frame_4, frame_5]
        elif frames_count < 5:
            for i in range(frames_count):
                mem_frame_ids.append(frame_ids[i])
            for i in range(5-frames_count):
                mem_frame_ids.append(frame_ids[len(frame_ids)-1])
        return mem_frame_ids


    ##### Get needed pseudo label frame ids ####
    def get_pseudo_lable(self, ann_id, frame_ids, epoch):
        assert ann_id
        with open('pseudo_lable_{}.json'.format(epoch-1), 'r') as pseudo_file:
            pseudo_dict = json.load(pseudo_file)
        frames_count = len(frame_ids)
        use_frame_ids = []
        if ann_id in pseudo_dict.keys():
            pseudo_frame = pseudo_dict[ann_id]
            if frames_count >= 2:
                n1 = random.sample(range(0,frames_count-1), 1)[0]
                n2 = random.sample(range(n1+1, min(frames_count,n1+1+self.skip)),1)[0]
                n3 = pseudo_frame[0]
                n4 = pseudo_frame[1]
                n5 = pseudo_frame[2]
                use_frame_ids.append(frame_ids[n1])
                use_frame_ids.append(frame_ids[n2])
                if n3 < frames_count:
                    use_frame_ids.append(frame_ids[n3])
                    # print('yes')
                else:
                    use_frame_ids.append(frame_ids[0])
                if n4 < frames_count:
                    use_frame_ids.append(frame_ids[n4])
                    # print('yes')
                else:
                    use_frame_ids.append(frame_ids[0])
                if n5 < frames_count:
                    use_frame_ids.append(frame_ids[n5])
                    # print('yes')
                else:
                    use_frame_ids.append(frame_ids[frames_count - 1])
            elif frames_count < 2:
                for i in range(5):
                    use_frame_ids.append(frame_ids[frames_count-1])  
        else:
            if frames_count >= 5:
                n1 = random.sample(range(0,frames_count-4), 1)[0]
                n2 = random.sample(range(n1+1, min(frames_count-3,n1+3+self.skip)),1)[0]
                n3 = random.sample(range(n2+1, min(frames_count-2,n2+2+self.skip)),1)[0]
                n4 = random.sample(range(n3+1, min(frames_count-1,n3+1+self.skip)),1)[0]
                n5 = random.sample(range(n4+1, min(frames_count,n4+self.skip)),1)[0]
                use_frame_ids.append(frame_ids[n1])
                use_frame_ids.append(frame_ids[n2])
                use_frame_ids.append(frame_ids[n3])
                use_frame_ids.append(frame_ids[n4])
                use_frame_ids.append(frame_ids[n5])
            elif frames_count < 5:
                for i in range(frames_count):
                    use_frame_ids.append(frame_ids[i])
                for i in range(5 - frames_count):
                    use_frame_ids.append(frame_ids[frames_count-1])
                    
        return use_frame_ids
    
    
    def get_iou_pseudo(self, ann_id, frame_ids, epoch):
        assert ann_id
        with open('/home/imi1214/MJP/projects/Refer-Youtube-VOS-NEW/checkpoint/refer-yv-2019/model/evaluation/refer-yv-2019_forward/e000{}.json'.format(epoch)) as evaluation_flie:
            evaluation_data = json.load(evaluation_flie)
        frames_count = len(frame_ids)   
        use_frame_ids = []
        exp_ids = list(evaluation_data["ious"].keys())
        # print(ann_id)
        if ann_id in evaluation_data["ious"].keys():
            # print(evaluation_data["ious"][ann_id])
            if frames_count >= 3:
                scores_list = evaluation_data["ious"][ann_id]
                index_list = [i[0] for i in sorted(enumerate(scores_list), key=lambda x:x[1])]
                # print(index_list)
                pseudo_frame = index_list[:3]
                n1 = random.sample(range(0,frames_count-1), 1)[0]
                n2 = random.sample(range(n1+1, min(frames_count,n1+1+self.skip)),1)[0]
                n3 = pseudo_frame[0]
                n4 = pseudo_frame[1]
                n5 = pseudo_frame[2]
                use_frame_ids.append(frame_ids[n1])
                use_frame_ids.append(frame_ids[n2])
                if n3 < frames_count:
                    use_frame_ids.append(frame_ids[n3])
                    # print('yes')
                else:
                    use_frame_ids.append(frame_ids[0])
                if n4 < frames_count:
                    use_frame_ids.append(frame_ids[n4])
                    # print('yes')
                else:
                    use_frame_ids.append(frame_ids[0])
                if n5 < frames_count:
                    use_frame_ids.append(frame_ids[n5])
                    # print('yes')
                else:
                    use_frame_ids.append(frame_ids[frames_count - 1])
            elif frames_count < 3:
                for i in range(5):
                    use_frame_ids.append(frame_ids[frames_count-1])  
        else:
            if frames_count >= 5:
                n1 = random.sample(range(0,frames_count-4), 1)[0]
                n2 = random.sample(range(n1+1, min(frames_count-3,n1+3+self.skip)),1)[0]
                n3 = random.sample(range(n2+1, min(frames_count-2,n2+2+self.skip)),1)[0]
                n4 = random.sample(range(n3+1, min(frames_count-1,n3+1+self.skip)),1)[0]
                n5 = random.sample(range(n4+1, min(frames_count,n4+self.skip)),1)[0]
                use_frame_ids.append(frame_ids[n1])
                
                use_frame_ids.append(frame_ids[n2])
                use_frame_ids.append(frame_ids[n3])
                use_frame_ids.append(frame_ids[n4])
                use_frame_ids.append(frame_ids[n5])
            elif frames_count < 5:
                for i in range(frames_count):
                    use_frame_ids.append(frame_ids[i])
                for i in range(5 - frames_count):
                    use_frame_ids.append(frame_ids[frames_count-1])
                    
        return use_frame_ids  
        
        
    ##### Origin frame selection #####
    def origin_frame_chose(self, frame_ids):
        num_frames = len(frame_ids)
        use_index = []
        mem_frame_ids = []
        if num_frames == 6:
            for i in range(num_frames):
                mem_frame_ids.append(frame_ids[i])
        elif num_frames < 6:
            for i in range(num_frames):
                mem_frame_ids.append(frame_ids[i])
            for i in range(6-num_frames):
                mem_frame_ids.append(frame_ids[len(frame_ids)-1])
        else:    
            id_1 = 0
            id_2 = 1
            id_6 = num_frames-1
            id_4 = num_frames//2
            id_3 = id_4 - id_4//2
            id_5 = id_4 + id_4//2
            use_index.append(id_1)
            use_index.append(id_2)
            use_index.append(id_3)
            use_index.append(id_4)
            use_index.append(id_5)
            use_index.append(id_6)
            for i in use_index:
                mem_frame_ids.append(frame_ids[i])
        return mem_frame_ids
       

    def __getitem__(self, index):
        
        if self.mode == 'eval_yv':
            vid, oid, category, frame_ids, sent, exp_id, _ = self.videos[index]
            
            Fs, Ms, initial_size = self.load_valid(vid, oid, frame_ids)   # The corresponding frame and mask of a target are obtained
            
            num_frames = len(Fs)
            # num_masks = len(Ms)
            
            if num_frames < self.max_frames:
                pad_frames = self.max_frames-num_frames
                # pad_masks = self.max_frames-num_masks
                Fs = F.pad(Fs, (0,0)*3+(0,pad_frames))  # (T,C,H,W) -> (48,C,H,W)
                # Ms = F.pad(Ms, (0,0)*2+(0,pad_masks))  # need to modify
            # Ms = F.pad(Ms, (0,0)*2+(0,self.max_frames - 1))
            # if num_masks < self.max_frames:
            #     pad_masks = self.max_frames-num_masks
            #     Ms = F.pad(Ms, (0,0)*2+(0,pad_masks))
            
            ann_id = '{}_{}'.format(vid, exp_id)
            inputs = self.tokenizer(sent, padding="max_length", truncation=True, return_tensors="pt", max_length=20)
            outputs = self.text_model(**inputs)
            words = outputs.last_hidden_state.detach()  # (1,512,768)
            words = torch.squeeze(words, dim=0)
            
            meta = {'sent': sent}
            
            if Fs is None:
                print("error")
            
            return Fs, Ms, words, ann_id, num_frames, frame_ids, meta, initial_size
        
        elif self.mode == 'eval_davis':
            vid, oid, category, frame_ids, sent, _ = self.videos[index]
            
            Fs, Ms = self.load_pairs(vid, oid, frame_ids)   # The corresponding frame and mask of a target are obtained
            
            num_frames = len(Fs)
            num_masks = len(Ms)
            
            inputs = self.tokenizer(sent, padding="max_length", truncation=True, return_tensors="pt", max_length=20)
            outputs = self.text_model(**inputs)
            words = outputs.last_hidden_state.detach()  # (1,512,768)
            words = torch.squeeze(words, dim=0)
            
            ann_id = '{}_{}'.format(vid, oid)
            
            meta = {'sent': sent}
            return Fs, Ms, words, ann_id, num_frames, frame_ids, meta  
        
        elif self.mode == 'train_davis':
            rnd = random.Random()
            vid, oid, category, frame_ids, sent, _ = self.videos[index]
            
            ann_id = '{}_{}'.format(vid, oid)
            
            epoch_num = get_epoch_number()
            
            if epoch_num > 20 and epoch_num <30:
                use_frame_ids = self.get_pseudo_lable(ann_id, frame_ids, epoch_num)
            else:
                use_frame_ids = self.sample_frame_ids_base(frame_ids)
            # use_frame_ids = self.sample_frame_ids_base(frame_ids)
            
            frames, masks = [], []
            for frame_id in use_frame_ids:
                frm, msk = self.load_pair(vid, oid, frame_id)
                if self.jitter:
                    frm, msk = self.random_jitter(frm, msk, self.size, rnd)
                else:
                    frm, msk = self.resize(frm, msk, self.size)
                frames.append(frm)
                masks.append(msk)
            
            frames = np.stack(frames, axis=0)
            masks = np.stack(masks, axis=0)
            
            Fs = torch.from_numpy(np.transpose(frames, (0, 3, 1, 2)).copy()).float()    # （num_frames, C, H, W）
            Ms = torch.from_numpy(masks.copy()).float()
            
            num_frames = len(Fs)    # 会有重复帧
            num_masks = len(Ms)
            
            # if num_frames < self.max_frames:    # 补充
            #     pad_frames = self.max_frames-num_frames
            #     Fs = F.pad(Fs, (0,0)*3+(0,pad_frames))
            #     Ms = F.pad(Ms, (0,0)*2+(0,pad_frames)) 
            
            inputs = self.tokenizer(sent, padding="max_length", truncation=True, return_tensors="pt", max_length=20)
            outputs = self.text_model(**inputs)
            words = outputs.last_hidden_state.detach()  # (1,512,768)
            words = torch.squeeze(words, dim=0)
            
            meta = {'sent': sent}
            return Fs, Ms, words, ann_id  
        
        elif self.mode == 'train_yv':
            rnd = random.Random()

            vid, oid, frame_ids, sent, exp_id, num_objects = self.videos[index]
            ann_id = '{}_{}'.format(vid, exp_id)
            epoch_num = get_epoch_number()
            
            if epoch_num > 1:
                use_frame_ids = self.get_pseudo_lable(ann_id,frame_ids,epoch_num)
            else:
                use_frame_ids = self.sample_frame_ids_base(frame_ids)

            # get frames and masks
            frames, masks = [], []
            for frame_id in use_frame_ids:
                frm, msk = self.load_pair(vid, oid, frame_id)
                if self.jitter:
                    frm, msk = self.random_jitter(frm, msk, self.size, rnd)
                else:
                    frm, msk = self.resize(frm, msk, self.size)
                frames.append(frm)
                masks.append(msk)
                
            frames = np.stack(frames, axis=0)
            masks = np.stack(masks, axis=0)

            Fs = torch.from_numpy(np.transpose(frames, (0, 3, 1, 2)).copy()).float()    # num_frames, C, H, W）
            Ms = torch.from_numpy(masks.copy()).float()
            # words = self.tokenize_sent(sent)
            # sent = self.tokenizer.encode(sent, padding="max_length", max_length=20)
            inputs = self.tokenizer(sent, padding="max_length", truncation=True, return_tensors="pt", max_length=20)
            outputs = self.text_model(**inputs)
            words = outputs.last_hidden_state.detach()  # (1,512,768)
            words = torch.squeeze(words, dim=0)
            
            return Fs, Ms, words, ann_id
        
        elif self.mode =='eval_yv_forward':
            
            vid, oid, frame_ids, sent, exp_id, num_frames = self.videos[index]
            
            Fs, Ms = self.load_pairs(vid, oid, frame_ids)   # 得到某一目标对应帧和掩码
            
            num_frames = len(Fs)    # 会有重复帧
            num_masks = len(Ms)

            if num_frames < self.max_frames:    # 补充
                pad_frames = self.max_frames-num_frames
                pad_masks = self.max_frames-num_masks
                Fs = F.pad(Fs, (0,0)*3+(0,pad_frames))
                Ms = F.pad(Ms, (0,0)*2+(0,pad_masks))

            inputs = self.tokenizer(sent, padding="max_length", truncation=True, return_tensors="pt", max_length=20)
            outputs = self.text_model(**inputs)
            words = outputs.last_hidden_state.detach()  # (1,512,768)
            words = torch.squeeze(words, dim=0)
            
            ann_id = '{}_{}'.format(vid, exp_id)
            
            meta = {'sent': sent}
            
            return Fs, Ms, words, ann_id, num_frames, num_masks, meta
        
        elif self.mode == 'eval_davis_forward':
            vid, oid, category, frame_ids, sent, _ = self.videos[index]
            
            ann_id = '{}_{}'.format(vid, oid)
            
            Fs, Ms = self.load_pairs(vid, oid, frame_ids)   # 得到某一目标对应帧和掩码
            
            num_frames = len(Fs)    # 会有重复帧
            num_masks = len(Ms)
            
            # if num_frames < self.max_frames:    # 补充
            #     pad_frames = self.max_frames-num_frames
            #     Fs = F.pad(Fs, (0,0)*3+(0,pad_frames))
            #     Ms = F.pad(Ms, (0,0)*2+(0,pad_frames)) 
            
            inputs = self.tokenizer(sent, padding="max_length", truncation=True, return_tensors="pt", max_length=20)
            outputs = self.text_model(**inputs)
            words = outputs.last_hidden_state.detach()  # (1,512,768)
            words = torch.squeeze(words, dim=0)
            
            meta = {'sent': sent}
            return Fs, Ms, words, ann_id  
        else:
            raise ValueError
    
    ##### tokenize sentences #####
    def tokenize_sent(self, sent):
        return self.corpus.tokenize(sent, self.query_len)


    ##### untokenize sentences #####
    def untokenize_word_vector(self, words):
        return self.corpus.untokenize(words)

    
    