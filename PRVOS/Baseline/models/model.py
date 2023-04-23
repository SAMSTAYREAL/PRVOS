from __future__ import division
from pyexpat import model
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
from torchvision import models
 
# general libs
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import math
import time
import tqdm
import os, pdb
import argparse
import copy
import sys

from resnest.torch import resnest101
 
sys.path.insert(0, '.')
# from common import *
from models.common import *
sys.path.insert(0, '../utils/')
from utils.helpers import *
# from common_SA import *

        
class Encoder_Q(nn.Module):
    def __init__(self,backbone):
        super(Encoder_Q, self).__init__()

        if backbone == 'resnet50':
            resnet = models.resnet50(pretrained=True)
        elif backbone == 'resnet18':
            resnet = models.resnet18(pretrained=True)
        elif backbone == 'resnest101':
            resnet = resnest101()
        
        # resnet = models.resnet50(pretrained=True)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu  # 1/2, 64
        self.maxpool = resnet.maxpool

        self.res2 = resnet.layer1 # 1/4, 256
        self.res3 = resnet.layer2 # 1/8, 512
        self.res4 = resnet.layer3 # 1/16, 1024
        self.res5 = resnet.layer4 # 1/32, 2048

        ####################
        # freeze_BN(self)
        ####################

        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def forward(self, frames): # B,T,C,H,W
        B,C,H,W = frames.size()

        f = (frames - self.mean) / self.std

        x = self.conv1(f)
        x = self.bn1(x)
        c1 = self.relu(x)   # 1/2, 64
        x = self.maxpool(c1)  # 1/4, 64
        r2 = self.res2(x)   # 1/4, 256
        r3 = self.res3(r2) # 1/8, 512
        r4 = self.res4(r3) # 1/16, 1024, (B, C, H, W),(2,1024,20,20)
        r5 = self.res5(r4)  # 1/32, 2048, (2,2048,10,10)
        
        return r5, r4, r3, r2, c1, f
    
 
# Memory encoder use the feature of the layer 4 
class Encoder_M(nn.Module):
    def __init__(self,backbone):
        super(Encoder_M, self).__init__()
        if backbone == 'resnest101':
            self.conv1_m = nn.Conv2d(1, 128, kernel_size=7, stride=2, padding=3, bias=False)
            self.conv1_o = nn.Conv2d(1, 128, kernel_size=7, stride=2, padding=3, bias=False)
        else:
            self.conv1_m = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.conv1_o = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        if backbone == 'resnet50':
            resnet = models.resnet50(pretrained=True)
        elif backbone == 'resnet18':
            resnet = models.resnet18(pretrained=True)
        elif backbone == 'resnest101':
            resnet = resnest101()

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu  # 1/2, 64
        self.maxpool = resnet.maxpool

        self.res2 = resnet.layer1 # 1/4, 256
        self.res3 = resnet.layer2 # 1/8, 512
        self.res4 = resnet.layer3 # 1/8, 1024

        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def forward(self, in_f, in_m):
        f = (in_f - self.mean) / self.std   # (B,3,3,384,384)
        m = torch.unsqueeze(in_m, dim=1).float() # add channel dim, (B,1,3,384,384), (B,C,T,H,W)

        x = self.conv1(f) + self.conv1_m(m) 
        x = self.bn1(x)
        
        c1 = self.relu(x)   # 1/2, 64
        x = self.maxpool(c1)  # 1/4, 64
        r2 = self.res2(x)   # 1/4, 256
        r3 = self.res3(r2) # 1/8, 512
        r4 = self.res4(r3) # 1/16, 1024
        return r4, r3, r2, c1, f
   
class MemAtt(nn.Module):
    def __init__(self, vis_q=1024, vis_m=1024):
        super(MemAtt, self).__init__()
 
    def forward(self, m_in, m_out, q_in, q_out):  # m_in: b,t,c,h,w， 
        # m_in:keys[0,1:um_objects+1], m_out:values[0,1:num_objects+1], q_in:k4e, q_out:v4e
        # k4e和v4e是target visual features的key和value
        # keys[0,1:um_objects+1], values[0,1:num_objects+1]是memory visual features的key和value
        B, T, D_e, H, W = m_in.size()
        _, _, D_o, _, _ = m_out.size()

        mi = m_in.view(B, D_e, T*H*W) 
        mi = torch.transpose(mi, 1, 2)  # b, THW, emb, 转换第二三维度
 
        qi = q_in.view(B, D_e, H*W)  # b, emb, HW
        
        # target frame的key和memory frame的key进行attention
        p = torch.bmm(mi, qi) # b, THW, HW
        p = p / math.sqrt(D_e)
        p = F.softmax(p, dim=1) # b, THW, HW

        # attention输出和memory frame的value进行矩阵乘法
        mo = m_out.view(B, D_o, T*H*W) 
        mem = torch.bmm(mo, p) # Weighted-sum B, D_o, HW
        mem = mem.view(B, D_o, H, W)

        mem_out = torch.cat([mem, q_out], dim=1)    # (B, C, H, W), (B, 512, H, W)

        return mem_out, p   # output memory attentive feature map M^
    

class CrossAtt(nn.Module):
    def __init__(self, vis_dim=2048, lang_dim=768, head_num=8, emb=512):
        super(CrossAtt, self).__init__()
    
        self.convSA = ConvSA(vis_dim, emb)  # 实例化ConvSA, C=vis_dim(2048), emb=emb(512)
        self.linearSA = LinearSA(lang_dim, emb)
        C = emb+emb # 1024
        
        self.Query = nn.Conv2d(C, emb, kernel_size=(3,3), padding=(1,1), stride=1)
        self.Key = nn.Conv2d(C, emb, kernel_size=(3,3), padding=(1,1), stride=1)
        self.Value = nn.Conv2d(C, emb, kernel_size=(3,3), padding=(1,1), stride=1)
        self.resValue = nn.Conv2d(C, emb, kernel_size=(3,3), padding=(1,1), stride=1)
        
        init_He(self)
        self.head_num = head_num
        
 
    def forward(self, vis, emb):  # vis: (B,C,H,W), emb: (B, L, C')
        
        B, C_v, H, W = vis.size()
        B, L, C_l = emb.size()
        
        vis_att = self.convSA(vis) # B,emb,H,W
        lang_att = self.linearSA(emb).transpose(1, 2)  # B,emb,L
        
        vis_ = vis_att.unsqueeze(2).repeat(1, 1, L, 1, 1)   # 在第3维度扩展扩充维度L倍
        emb_ = lang_att.unsqueeze(3).unsqueeze(3).repeat(1, 1, 1, H, W)
        multi = torch.cat((vis_, emb_), 1) # B,C',L,H,W
        multi = multi.transpose(1, 2).reshape(B*L, -1, H, W)
        
        query = self.Query(multi).view(B, L, -1, H, W)  # (1, L, 512, H, W)
        key = self.Key(multi).view(B, L, -1, H, W)
        value = self.Value(multi).view(B, L, -1, H, W)
        res_value = self.resValue(multi).view(B, L, -1, H, W)
        
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        res_value = res_value.transpose(1, 2)
        
        query_ = query.reshape(B*self.head_num, -1, L*H*W) # (B*8, C, L*H*W)
        key_ = key.reshape(B*self.head_num, -1, L*H*W)
        value_ = value.reshape(B*self.head_num, -1, L*H*W)
        
        att = torch.bmm(query_.transpose(1, 2), key_)   # torch.bmm() 矩阵乘法, att:(B*8, L*H*W, L*H*W)
        att = F.softmax(att, dim=2) # (B*8, L*H*W, L*H*W)
        
        v_att = torch.bmm(value_, att.transpose(1, 2))  # (B*8, C, L*H*W)
        v_att = v_att.reshape(B, -1, L, H, W)
        
        value_att = torch.mean(v_att + res_value, 2)    # (B, C, L, H, W)
        return value_att # output cross-modal attentive map C^
          
        
class Memorize(nn.Module):
    def __init__(self, backbone = 'resnet50'):
        super(Memorize, self).__init__()
        self.backbone = backbone
        self.Encoder_M = Encoder_M(backbone)
        scale_rate = (1 if (backbone == 'resnet50' or backbone == 'resnest101') else 4)
        self.KV_M_r4 = KeyValue(1024//scale_rate, keydim=128//scale_rate, valdim=512//scale_rate)   # indim = 1024, keydim = 128, valdim = 512
    
            
    def forward(self, frames, masks):
        # _, T, H, W = masks.shape    # (B,T,H,W), (B,T,C,H,W)
        # pdb.set_trace()
        (frames, masks), pad = pad_divide_by([frames, masks], 16, (frames.size()[2], frames.size()[3]))
        
        # make batch arg list
        B_list = {'f':[], 'm':[]}
        # for f in range(3):
        B_list['f'].append(frames)
        B_list['m'].append(masks)
            
        # make batch
        B_ = {}
        for arg in B_list.keys():
            B_[arg] = torch.cat(B_list[arg], dim=0)
        
        r4, _, _, _, _ = self.Encoder_M(B_['f'], B_['m'])   
        k4, v4 = self.KV_M_r4(r4) # 128 and 512, H/16, W/16, ()
        k4e = torch.unsqueeze(k4, dim=1).float()
        v4e = torch.unsqueeze(v4, dim=1).float()
        return k4e, v4e        # k4:(B,T,C,H,W), (4,1,128,20,20), v4:(4,1,512,20,20)


class Query(nn.Module):
    def __init__(self, backbone = 'reasnet50'):
        super(Query, self).__init__()
        self.backbone = backbone
        self.Encoder_Q = Encoder_Q(backbone)
        scale_rate = (1 if (backbone == 'resnet50' or backbone == 'resnest101') else 4)
        self.KV_Q_r4 = KeyValue(1024//scale_rate, keydim=128//scale_rate, valdim=512//scale_rate)
        
    def forward(self, frame):
        # _, keydim, T, H, W = keys.shape # B = (B,3,128,1,24,24), (B,C,T,H,W)
        #pad
        [frame], pad = pad_divide_by([frame], 16, (frame.size()[2], frame.size()[3]))
        
        _, r4, r3, r2, _, _ = self.Encoder_Q(frame)
        k4, v4 = self.KV_Q_r4(r4)   #(1,128,24,24),(1,512,24,24)
        
        return k4, v4
        

class Decoder(nn.Module):
    def __init__(self, mdim, multi_dim, scale_rate, backbone):  # mdim=256, multi_dim=512
        super(Decoder, self).__init__()
        self.backbone = backbone
        self.backbone == 'resnest101'
        if backbone == 'resnest101':
            self.convFM = nn.Conv2d(256, mdim, kernel_size=(3,3), padding=(1,1), stride=1)
        else:
            self.convFM = nn.Conv2d(1024//scale_rate, mdim, kernel_size=(3,3), padding=(1,1), stride=1) # in_channels=1024, out_channels=256
        self.res_multi = nn.Sequential(
            nn.Conv2d(multi_dim, mdim, kernel_size=(3,3), padding=(1,1), stride=1),     # in_channels=512, out_channels=256
            ResBlock(mdim, mdim)
        )
        
        self.Mem_emb = nn.Conv2d(512, mdim, kernel_size=(3,3), padding=(1,1), stride=1)
        self.RF4 = Refine(1024, mdim) # mdim = 256
        self.RF3 = Refine(512, mdim) 
        
        self.RF2 = Refine(256, mdim) 
        self.self_attention = Self_Attention(mdim)
        # self.pred2 = nn.Conv2d(mdim, 2, kernel_size=(3,3), padding=(1,1), stride=1)

    # multi5为cross-attention输出， mem4为memory-attention输出
    def forward(self, multi5, mem4, v5, v4, v3, v2, v1):
        # pdb.set_trace()
        # with torch.no_grad():  ###############################################################
        m5 = self.res_multi(multi5) # indim = 512, outdim = 256, multi5:(B,512,20,20), m5:(2,256,10,10)
        # with torch.no_grad():
        r4 = self.RF4(v4 + mem4, m5)   # indim = 1024, outdim = 256, v4:(2,1024,20,20), mem4:(2,1024,20,20)
        r3 = self.RF3(v3, r4)   # indim = 512, outdim = 256, v3:(2,512,40,40), r4:(2,256,20,20)
        r2 = self.RF2(v2, r3)   # indim = 256, outdim = 256, v2:(2,256,80,80), r3:(2,256,40,40)
        logit = self.self_attention(m5, r4, r3, r2)
        
        pred = F.interpolate(logit, scale_factor=4, mode='bilinear', align_corners=False)  # upsample 
        
        return pred # (2,256,320,320)
        

class URVOS(nn.Module):
    def __init__(self, backbone = 'resnet50'):
        super(URVOS, self).__init__()
        # with torch.no_grad():
        self.backbone = backbone
        assert backbone == 'resnet50' or backbone == 'resnet18' or backbone == 'resnest101'
        scale_rate = (1 if (backbone == 'resnet50' or backbone == 'resnest101') else 4)
        
        self.Encoder_M = Encoder_M(backbone)
        self.Encoder_Q = Encoder_Q(backbone)
        
        self.query = Query(backbone)
        # self.memorize = nn.ModuleList([Memorize(backbone)])
        self.memorize = Memorize(backbone)
        # self.embs = nn.Embedding(dict_size, 1000)   ##################
        self.cas = CrossAtt(vis_dim=2048, lang_dim=768)   # cross-attention
        self.mas = MemAtt(vis_q=1024, vis_m=1024)
        
        self.KV_M_r4 = KeyValue(1024//scale_rate, keydim=128//scale_rate, valdim=512//scale_rate)
        self.KV_Q_r4 = KeyValue(1024//scale_rate, keydim=128//scale_rate, valdim=512//scale_rate)
        
        self.Decoder = Decoder(256,512,scale_rate,backbone)
        
    def forward(self, mem_frames, mem_masks, in_frames, words, eval=False):
        
        # _, T, H, W = mem_masks.shape  # (B,T.H,W)
        # pdb.set_trace()
        # embed = self.embs(words)
        # print(words.size())
        with torch.no_grad():
            vis_r5s, vis_r4s, vis_r3s, vis_r2s, vis_c1s,_ = self.Encoder_Q(in_frames)
            # with torch.no_grad():
            # pdb.set_trace()
            k_m, v_m = self.memorize(mem_frames, mem_masks) # (4,1,128,20,20), (4,1,512,20,20)
            k_q, v_q = self.query(in_frames)    # in_frames:(4,3,320,320), k_q:(4,128,20,20), v_q:(4,512,20,20)
            mem_r4, p = self.mas(k_m, v_m, k_q, v_q)    # mem_r4:(2,1024,20,20)
            multi_r5s = self.cas(vis_r5s, words)    # multi_r5s:(2,512,10,10)
        logit = self.Decoder(multi_r5s, mem_r4, vis_r5s, vis_r4s, vis_r3s, vis_r2s, vis_c1s)    # (2,256,320,320)
        # pdb.set_trace()
        mask = F.softmax(logit, dim=1)  # (2,256,320,320)
        
        return mask, logit