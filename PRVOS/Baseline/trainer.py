# -*- coding: utf-8 -*-
from __future__ import division
import sys
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
# import warnings
# warnings.simplefilter("ignore", UserWarning)

import torch
import torch.nn as nn
from torch.utils import data
import torch.utils.model_zoo as model_zoo
from torchvision import models

# general libs
import cv2, pdb
from PIL import Image
import numpy as np
from tqdm import tqdm
from addict import Dict
import json
import pickle

import os, sys, logging, time, random, json
from pathlib import Path
from shutil import copyfile

import math

### My libs
sys.path.append('utils/')
sys.path.append('models/')
sys.path.append('dataset/')

from utils.helpers import *

# from io_utils import *
from utils.io_utils import *
# from eval_utils import *
from utils.eval_utils import *
from tools.handle_json.make_label import make_json
from utils.pseudo_label_utils import *
from tools.handle_json.array_into_json import NumpyEncoder
from utils.model_ema import EMA

import dataset.factory as factory

from dataset.refer_datasets import REFER_YV_2019

from tools.load_pretrained_weights import load_trained_model_to_fintune

# DATA_ROOT = Path('./data')
DATA_ROOT = '/home/imi1214/MJP/datasets/RVOS/refer-yv-2019'
DATA_ROOT_DAVIS = '/home/imi1214/MJP/datasets/RVOS/Ref-DAVIS'
CHECKPOINT_ROOT = Path('./checkpoint')
OUTPUT_IMG_ROOT = Path('./validation')
EVALUTAION_ROOT = Path('./evaluation')

class Trainer():

    def __init__(self, args):
        import importlib
        
        self.args = args
        
        self.init_lr = args.init_lr if args.init_lr else 1e-4   # initial learning rate
        self.lr = 0
        
        self.epoch = -1
        self.max_epoch = args.max_epoch
        self.decay_epochs = args.decay_epochs
        self.lr_decay = args.lr_decay   # 0.1
        self.save_every = 1 if not args.save_every else args.save_every # 0
        
        self.img_size = (args.img_size, args.img_size)
        self.batch_size = args.batch_size
        self.test_batch_size = args.test_batch_size if args.test_batch_size else args.batch_size
        self.max_N = 2 if not args.max_N else args.max_N
        self.max_skip = args.max_skip
        
        ######## ref-yv-2019 evaluation output files #####
        self.checkpoint = args.epoch
        ##################################################

        self.desc = args.desc
        # self.arch = args.arch
        self.arch = 'model'
        self.splits = args.splits
        self.mode = args.mode
        self.finetune = args.finetune
        # self.splits = '0.25'
        
        ###### Need to modify #####    
        self.model = importlib.import_module('models.{}'.format(self.arch)).URVOS()  # models.base_model.Mask()
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr)
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none')
        self.scheme = self.base_scheme
        
        self.ema = EMA(self.model, 0.999)
        
        
        self.logger = get_logger(self.arch)
        
        
    ##### Get file name #####
    def get_file_name(self):    
        file_name = self.arch
        if self.desc:
            file_name += '_' + self.desc
        return file_name
    
    
    ##### Get save path #####
    def get_save_path(self):
        file_name = self.get_file_name()
        save_path = CHECKPOINT_ROOT / self.dataset / file_name
        return save_path
            
            
    ##### GPU parallel #####
    def cuda(self):             
        # self.model = nn.DataParallel(self.model).cuda()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)
        self.criterion = self.criterion.cuda()
        
        
    ##### Refresh learning rate ####
    def update_hyperparam_epoch(self):  
        init_lr = self.init_lr
        self.N = self.max_N
        
        if len(self.decay_epochs) > 0:
            lr = init_lr
            for decay in self.decay_epochs:
                if self.epoch >= decay: 
                    lr = lr * self.lr_decay # 0.0001*0.1 = 0.00001
        
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr

                self.lr = lr


    ##### Load model ####
    def load_model(self, epoch=0):      
        if epoch==0:
            file_name = self.get_file_name()
            checkpoint_dir = CHECKPOINT_ROOT / self.dataset / file_name
            checkpoint_path = max((f.stat().st_mtime, f) for f in checkpoint_dir.glob('*.pth'))[1]
            self.logger.info('Resume Latest from {}'.format(checkpoint_path))
        else:
            self.logger.info('Resume from {}'.format(epoch))
            file_name = self.get_file_name()
            checkpoint_path = CHECKPOINT_ROOT / self.dataset / file_name / 'e{:04d}.pth'.format(epoch)
            
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if self.finetune:
            checkpoint = load_trained_model_to_fintune(checkpoint)  #### delete the parameters of Decoder directly
        self.model.load_state_dict((checkpoint['state_dict']), strict=False) # Set CUDA before if error occurs. strict=False
        # self.optimizer.load_state_dict(checkpoint['optimizer'])   ########
        self.epoch = checkpoint['epoch']

    
    ##### Save checkpoint ####
    def save_checkpoint(self):  
        save_path = self.get_save_path()
        save_file = 'e{:04d}.pth'.format(self.epoch+1)
        if not save_path.exists():
            save_path.mkdir(parents=True, exist_ok=True)
        
        torch.save({
                'epoch': self.epoch,
                'arch': self.arch,
                'state_dict': self.model.state_dict(),
                'optimizer' : self.optimizer.state_dict(),
                }, save_path / save_file )

        self.logger.info("Saved a checkpoint {}.".format(save_path / save_file))

    ######## training set ###########
    def set_trainset(self, dataset):
        self.dataset = dataset
        if dataset == 'refer-yv-2019':
            split_type = 'train_full'
            trainset = REFER_YV_2019(data_root=os.path.join(DATA_ROOT,'Youtube-VOS'), split=split_type, N=self.max_N,size=(320,320), mode='train_yv', jitter=True, scale=1.0)
            trainLoader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size, shuffle=True, num_workers=8, drop_last=True, pin_memory=True)
        elif dataset == 'ref-davis':
            split_type = 'train_full'
            trainset = REFER_YV_2019(DATA_ROOT_DAVIS, split=split_type, N=self.max_N,size=(320,320), mode='train_davis', jitter=True, scale=1.0)
            trainLoader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size, shuffle=True, num_workers=8, drop_last=True, pin_memory=True)
        self.train_set = trainset
        self.train_loader = trainLoader
    ######## validation set ###########   
    def set_valset(self, test_dataset):
        self.test_dataset = test_dataset
        if test_dataset == 'refer-yv-2019':
            split_type = 'valid'
            testset = REFER_YV_2019(data_root=os.path.join(DATA_ROOT,'Youtube-VOS'), split=split_type, size=(320,320), mode='eval_yv')
            testLoader = torch.utils.data.DataLoader(testset, batch_size=self.test_batch_size, shuffle=False, num_workers=4, drop_last=False, pin_memory=True)
        elif test_dataset == 'ref-davis':
            split_type = 'valid'
            testset = REFER_YV_2019(DATA_ROOT_DAVIS, split=split_type, size=(320,320), mode='eval_davis')
            testLoader = torch.utils.data.DataLoader(testset, batch_size=self.test_batch_size, shuffle=False, num_workers=4, drop_last=False, pin_memory=True)
        self.val_set = testset
        self.val_loader = testLoader
    ######## forward only set ###########
    def set_forwardset(self, dataset):
        if dataset == 'refer-yv-2019':
            split_type = 'train_full'
            forwardset = REFER_YV_2019(data_root=os.path.join(DATA_ROOT,'Youtube-VOS'), split=split_type, N=self.max_N,size=(320,320), mode='eval_yv_forward', jitter=True, scale=1.0)
            forwardLoader = torch.utils.data.DataLoader(forwardset, batch_size=self.batch_size, shuffle=True, num_workers=8, drop_last=True, pin_memory=True)
        elif dataset == 'ref-davis':
            split_type = 'train_full'
            forwardset = REFER_YV_2019(DATA_ROOT_DAVIS, split=split_type, N=self.max_N,size=(320,320), mode='eval_davis_forward', jitter=True, scale=1.0)
            forwardLoader = torch.utils.data.DataLoader(forwardset, batch_size=self.batch_size, shuffle=True, num_workers=8, drop_last=True, pin_memory=True)
        self.forward_set = forwardset
        self.forward_loader = forwardLoader

    ##### Scheme #####
    def base_scheme(self, frames, gt_masks, words, eval=False):
        # (2,5,3,320,320)
        if eval:
            B, T, _, W, H = frames.size()
            # print("T is {}".format(T))
            est_masks = torch.zeros_like(frames).sum(2)

            loss = 0.0
            prev_frame, prev_mask = None, None
            
            for t in range(0, T):
                
                if t == 0:
                    mask_pred, logit =\
                        self.model(frames[:,0], gt_masks[:,0], frames[:,1], words, eval=True)
                elif t == 0 and T == 1:
                    mask_pred, logit =\
                        self.model(frames[:,0], gt_masks[:,0], frames[:,0], words, eval=True)
                else:
                    mask_pred, logit =\
                        self.model(prev_frame, prev_mask, frames[:,t], words, eval=True)

                loss += torch.mean(self.criterion(logit, gt_masks[:,0].long())) ###### eval_yv  gt_masks[:,0]

                prev_frame, prev_mask = frames[:, t], mask_pred[:, 1]
                est_masks[:,t] = mask_pred[:,1].detach()
                
            return est_masks, loss, 0, T
                
            
        N = frames.size(1)
        # print("frames size is {}".format(N))
        est_masks = torch.zeros_like(gt_masks)
        loss = 0.0
        L2 = 0.0
        lambda_loss = 10

        prev_frame, prev_mask = None, None 

        for n in range(0, N-1):
            if n == 0:
                mask_pred, logit =\
                    self.model(frames[:,0], gt_masks[:,0], frames[:,1], words, eval=False)
                L1 = torch.mean(self.criterion(logit, gt_masks[:,n].long()))
            else:
                mask_pred, logit =\
                        self.model(prev_frame, prev_mask, frames[:,n], words, eval=True)    # modified
                L2 += torch.mean(self.criterion(logit, gt_masks[:,n].long()))
            
            # loss += torch.mean(self.criterion(logit, gt_masks[:,n].long()))

            prev_frame = frames[:, n]   # (1,3,320,320)
            prev_mask = mask_pred[:, 1]  # (1,320,320)
            # prev_mask = gt_masks[:, n]
            est_masks[:,n] = mask_pred[:,1].detach() # cut grad, (1,320,320)
            
        loss = L1 + lambda_loss*L2
        
        return est_masks, loss, 0, N-1
      
    # Training        
    def train(self):
        self.epoch += 1
        
        for self.epoch in range(self.epoch, self.max_epoch):
            
            self.update_hyperparam_epoch()
            self.logger.info('=========== EPOCH {} | LR {}  | N {} |  training=========='.format(self.epoch+1, self.lr, self.N))

            batch_time = AverageMeter('Time', ':6.3f')
            data_time = AverageMeter('Data', ':6.3f')
            losses = AverageMeter('Loss', ':2.4f')
            IoUs = [AverageMeter('IoU_{}'.format(i), ':3.4f') for i in range(4)]
            mIoU = AverageMeter('mIoU', ':3.4f')
            J = AverageMeter('J', ':3.4f')
            F = AverageMeter('F', ':3.4f')
            
            end = time.time()

            for i, V in enumerate(tqdm(self.train_loader, dynamic_ncols=True)):
                
                data_time.update(time.time() - end)
                self.ema.register()

                frames, gt_masks, words, ann_id = V  
                frames, gt_masks, words = ToCuda([frames, gt_masks, words])   # (1,3,3,320,320), (1,3,320,320)

                est_masks, loss, N_start, N_end = self.scheme(frames, gt_masks, words, eval=False)  # N_start = 0, N_end = 4
                
                losses.update(loss.item(), N_end - N_start)
                
                iou = [0]*(N_end - N_start)
                for n in range(N_start, N_end):
                    iou_n = IoU(est_masks[:,n], gt_masks[:,n])
                    # print(est_masks[:,n].shape) 
                    iou[n] = iou_n
                    
                miou = sum(iou[N_start:N_end])/(N_end - N_start)
              
                for n in range(N_start, N_end):
                    IoUs[n].update(iou[n])
                mIoU.update(miou)
                    
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.ema.update()
                
                batch_time.update(time.time() - end)
                end = time.time()
                
                if (i+1) % 10 == 0 or (i+1) == len(self.train_loader): 

                    self.logger.info('{} | E [{:d}] | I [{:d}] | {} | {} | {} | {}'.format(
                        self.arch, self.epoch+1, i+1, losses, mIoU, data_time, batch_time))
            
            # save a checkpoint
            save_every = self.save_every
            if self.epoch > self.decay_epochs[0]:
                save_every = min(self.save_every, 5)    
            
            self.forward_generate_pseudo()
           
            if (self.epoch + 1) % save_every == 0:
                self.save_checkpoint()
                del loss, frames, gt_masks, words, V, est_masks
                self.evaluate()
                torch.cuda.empty_cache()
            else:
                continue

    
    def forward_generate_pseudo(self):
        pseudo_lable = {}
        save_path = self.get_save_path()
        
        if self.mode == 'eval_yv_forward':
            eval_path = None
            split_name = '{}_forward'.format(self.dataset)
            eval_path = save_path / 'evaluation' / split_name
            if not eval_path.exists():
                eval_path.mkdir(parents=True, exist_ok=True)
        elif self.mode == 'eval_davis':
            eval_path = None
            split_name = '{}_{}'.format(self.dataset, self.splits)
            eval_path = save_path / 'evaluation' / split_name
            if not eval_path.exists():
                eval_path.mkdir(parents=True, exist_ok=True)
        else:
            eval_path = None
            split_name = '{}_forward'.format(self.dataset)
            eval_path = save_path / 'evaluation' / split_name
            if not eval_path.exists():
                eval_path.mkdir(parents=True, exist_ok=True)
                
        self.logger.info('========================================== GENERATING PSEUDO LABELS ================================================')  
        
        with torch.no_grad():
            for i, V in enumerate(tqdm(self.forward_loader, dynamic_ncols=True)):
                frames, gt_masks, words, ref_ids, num_frames, num_masks = V  
                
                Tf = num_frames.max().item()   # batch为1，所以不需要
                Tm = num_masks.max().item()
                
                frames, gt_masks = frames[:, :Tf] , gt_masks[:, :Tm]
                _, _, _, W, H = frames.size()
                (frames, gt_masks), pad = pad_divide_by([frames, gt_masks], 16, (W, H))
                
                frames, gt_masks, words = ToCuda([frames, gt_masks, words])   
                est_masks, loss, N_start, N_end = self.scheme(frames, gt_masks, words, eval=True)
                B_,_,_,_ = est_masks.size()
                entropy = - torch.sum(torch.sum(est_masks * torch.log(est_masks), dim=2), dim=2)
                value_entropy, indices = torch.sort(entropy,descending=False)
                for i in range(B_):
                    index_pseudo = indices[i,:3]
                    index_pseudo = index_pseudo.squeeze(0).cpu().numpy()
                    pseudo_lable[ref_ids[i]] = index_pseudo
                    
                    json_str = json.dumps(pseudo_lable, cls = NumpyEncoder,indent=4)
        
                    with open('pseudo_lable_{}.json'.format(self.epoch),'w') as file:
                        file.write(json_str)
            
            pseudo_lable = {}
            del V, frames, gt_masks, words
                
    # Evaluation on Ref-DAVIS17/valid or refer-yv-2019/train
    @torch.no_grad()           
    def evaluate(self):
        if self.ema is not None:
            print('ema yes')
            self.ema.apply_shadow()
        save_path = self.get_save_path()
        
        if self.mode == 'eval_yv_forwar':
            eval_path = None
            split_name = '{}_forward'.format(self.dataset)
            eval_path = save_path / 'evaluation' / split_name
            if not eval_path.exists():
                eval_path.mkdir(parents=True, exist_ok=True)
        elif self.mode == 'eval_davis':
            eval_path = None
            split_name = '{}_{}'.format(self.dataset, self.splits)
            eval_path = save_path / 'evaluation' / split_name
            if not eval_path.exists():
                eval_path.mkdir(parents=True, exist_ok=True)
        else:
            eval_path = None
            split_name = '{}_forward'.format(self.dataset)
            eval_path = save_path / 'evaluation' / split_name
            if not eval_path.exists():
                eval_path.mkdir(parents=True, exist_ok=True)

        eval_json = Dict()
        eval_json.arch = self.arch
        eval_json.epoch = self.epoch
        eval_json.dataset = self.dataset

        self.logger.info('=========== EVALUATE MODEL {} EPOCH {}  >>>>  {}/{} =========='.format(self.arch, self.epoch+1, self.test_dataset, self.mode))

        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':2.4f')
        
        J = AverageMeter('J', ':3.4f')
        F = AverageMeter('F', ':3.4f')
        
        end = time.time()
        
        precs_thres = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
        precs = np.zeros(len(precs_thres))
        num_samples = 0
        
        with torch.no_grad():
            for i, V in enumerate(tqdm(self.val_loader, dynamic_ncols=True)):
                
                data_time.update(time.time() - end)

                frames, gt_masks, words, ref_ids, num_frames, num_masks, metas = V 
                
                # Tf = num_frames.max().item()   
                # Tm = num_masks.max().item()
                
                # T = int(num_frames.detach().cpu().numpy())

                # frames, gt_masks = frames[:, :Tf] , gt_masks[:, :Tm]
                
                # B, T, _, W, H = frames.size()
                # (frames, gt_masks), pad = pad_divide_by([frames, gt_masks], 16, (W, H))
                frames, gt_masks, words = ToCuda([frames, gt_masks, words])

                est_masks, loss, N_start, N_end = self.scheme(frames, gt_masks, words, eval=True)
                # B,T,H,W = est_masks.size()
                losses.update(loss.item(), N_end - N_start)
                    
                for b, n_frame in enumerate(num_frames):
                    j_score = 0.
                    f_score = 0.
                    ious = []
                    for t in range(n_frame):
                        iou_t = IoU(est_masks[b:b+1,t], gt_masks[b:b+1,t])
                        j_score += iou_t
                        ious.append(iou_t)
                        f_score += db_eval_boundary(est_masks[b:b+1,t], gt_masks[b:b+1,t])

                    j_score /= float(n_frame)
                    f_score /= float(n_frame)
                    iou_all = IoU(est_masks[b], gt_masks[b])

                    J.update(j_score)
                    F.update(f_score)

                    eval_json.j_score[ref_ids[b]] = j_score
                    eval_json.f_score[ref_ids[b]] = f_score
                    eval_json.ious[ref_ids[b]] = ious                  
                    
                    # Precision
                    precs += (j_score>precs_thres).astype(int)
                    num_samples += 1
                
                    if (i+1) % 10 == 0: 

                        self.logger.info('{} | I [{:d}] | {} | {} | {} | {} | {} | {} '.format(
                            self.arch, i+1, losses, iou_all, J, F, data_time, batch_time))

                    batch_time.update(time.time() - end)
                    end = time.time()  
             
            precs /= num_samples

            eval_json.average_J = J.avg
            eval_json.average_F = F.avg

            eval_json.prec5 = precs[0]
            eval_json.prec6 = precs[1]
            eval_json.prec7 = precs[2]
            eval_json.prec8 = precs[3]
            eval_json.prec9 = precs[4]

            
            json.dump(eval_json, open(eval_path / 'e{:04d}.json'.format(self.epoch+1), 'w'))
            
            self.logger.warning('{} | E [{:d}] | {} | {} | {} | {} | {}'.format(
                self.arch, self.epoch+1, losses, J, F, data_time, batch_time,
            ))
            
            del V, frames, gt_masks, words, ref_ids, num_frames, num_masks, metas


    @torch.no_grad()
    def evaluate_refer_youtube_vos(self):
        
        output_path = os.path.join(DATA_ROOT,'results',"{}_{}".format(self.test_dataset, self.checkpoint))
        if not os.path.exists(output_path): 
            os.mkdir(output_path)
        
        for i, V in enumerate(tqdm(self.val_loader, dynamic_ncols=True)):
            frames, gt_masks, words, ref_ids, num_frames, frame_ids, metas, initial_size = V
            vid = ref_ids[0].split('_')[0]  
            eid = ref_ids[0].split('_')[1]
            print(ref_ids)
            
        
            video_dir = os.path.join(output_path, vid)
            if not os.path.exists(video_dir):
                os.mkdir(video_dir)

            exp_dir = os.path.join(video_dir, str(eid))
            if not os.path.exists(exp_dir):
                os.mkdir(exp_dir)
            
            T = num_frames.max().item()       
            frames = frames[:, :T]

            frames, gt_masks, words = ToCuda([frames, gt_masks, words])

            est_masks, loss, N_start, N_end = self.scheme(frames, gt_masks, words, eval=True)
        
            H, W = int(initial_size[0]), int(initial_size[1])

            for t in range(len(frame_ids)):
                prediction_mask = est_masks[:,t]
                prediction_mask = torch.squeeze(prediction_mask, dim=0)
                prediction_mask = prediction_mask.cpu().numpy() # transfer tensor to cpu, because numpy is 'cpu only'
                prediction_mask = np.uint8(255*prediction_mask)
                _,prediction_mask = cv2.threshold(prediction_mask,120,255,cv2.THRESH_BINARY)
                prediction_mask = cv2.resize(prediction_mask, (H, W))
                
                save_path = os.path.join(exp_dir, "{}.png".format(frame_ids[t][0]))
                cv2.imwrite(save_path, prediction_mask, [int(cv2.IMWRITE_PNG_COMPRESSION),9])
                        
                    
                    
                    
                    
                
            
            
        
        
            
            
    