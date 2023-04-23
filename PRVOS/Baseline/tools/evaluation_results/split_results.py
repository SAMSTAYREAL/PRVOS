########  Screen out the results of the valid set ##############
import json
import os
import os.path
import shutil
import cv2
import threading
import time
from PIL import Image
import exifread

meta_path = "/home/imi1214/MJP/datasets/RVOS/refer-yv-2019/Youtube-VOS/meta_expressions_test/meta_expressions/test/meta_expressions.json"
result_path = "/home/imi1214/MJP/datasets/RVOS/refer-yv-2019/results/refer-yv-2019_20"
test_path = "/home/imi1214/MJP/datasets/RVOS/refer-yv-2019/results/refer-yv-2019_20/test"
valid_path = "/home/imi1214/MJP/datasets/RVOS/refer-yv-2019/results/refer-yv-2019_20/valid"
valid_annotation_path = '/root/autodl-tmp/datasets/refer-yv-2019/Youtube-VOS/valid/Annotations'
valid_JPEG_path = '/root/autodl-tmp/datasets/refer-yv-2019/Youtube-VOS/valid/JPEGImages'
pseudo_label_path = '/root/autodl-tmp/code/Refer-Youtube-VOS-3090/pseudo_lable_18.json'

sample_path = "/home/imi1214/MJP/datasets/RVOS/refer-yv-2019/results/Annotations"

def read_json_vid(path:str):
    json_1 = None
    with open(path, 'r') as f:
        json_1 = json.load(f)
    ids = list(json_1["videos"].keys())
    return ids
def split_results(path):
    filelist = os.listdir(path)
    print(len(filelist))
    vids = read_json_vid(meta_path)
    for filename in filelist:
        if filename in vids:
            print("test video is；{}".format(filename))
            shutil.copytree(os.path.join(result_path, "{}".format(filename)), os.path.join(test_path, "{}".format(filename)))
            # shutil.rmtree(os.path.join(result_path, "{}".format(vid)))
        elif filename not in vids:
            print("valid video is {}".format(filename))
            shutil.copytree(os.path.join(result_path, "{}".format(filename)), os.path.join(valid_path, "{}".format(filename)))
            # shutil.rmtree(os.path.join(result_path, "{}".format(vid)))
        # print(filename)
        
def check_file_number(path):
    filelist = os.listdir(path)
    print(len(filelist))

def check_all_file_number(path):
    dir_count = 0
    file_count = 0
    for root,dirs,files in os.walk(path):    #遍历统计
        for dir in dirs:
            dir_count += 1 # 统计文件夹下的文件夹总个数
        for _ in files:
            file_count += 1   # 统计文件夹下的文件总个数
    print ('dir_count ', dir_count) # 输出结果
    print ('file_count ', file_count)

    
# file num error
def compare_to_sample(path1, path2):
    filelist_sample_vid = os.listdir(path1)
    filelist_valid_vid = os.listdir(path2)
    for vid_ in filelist_valid_vid:
        if vid_ in filelist_sample_vid:
            print('video ids is ok')
        else:
            print('there is no {} in sample file'.format(vid_))
            
    for vid in filelist_sample_vid:
        filelist_exp_1 = os.listdir(os.path.join(path1, vid))
        filelist_exp_2 = os.listdir(os.path.join(path2, vid))
        for exp in filelist_exp_1:
            if exp in filelist_exp_2:
                print('exp ids is ok')
            else:
                print('there is no {} in video {}'.format(exp, vid))
            
            if len(filelist_exp_1) == len(filelist_exp_2):
                filelist_frame_1 = os.listdir(os.path.join(path1, vid, exp))
                filelist_frame_2 = os.listdir(os.path.join(path2, vid, exp))
                if len(filelist_frame_1) == len(filelist_frame_2):
                    print('frame number is ok')
                    continue
                else:
                    print("wrong file is:{}_{}".format(vid, exp))
            else:
                print("lack of expression file:{}".format(vid))

def reduce_quality(path):
    filelist_valid_vid = os.listdir(path)
    for vid in filelist_valid_vid:
        filelist_exp = os.listdir(os.path.join(path, vid))
        for exp in filelist_exp:
            filelist_frame = os.listdir(os.path.join(path, vid, exp))
            # print(filelist_frame)
            for mask_name in filelist_frame:
                mask_path = os.path.join(path, vid, exp, mask_name)
                mask = Image.open(mask_path).convert('L')
                mask.save(mask_path)
                # mask = cv2.imread(mask_path)
                # cv2.imwrite(mask_path, mask, [int(cv2.IMWRITE_PNG_COMPRESSION),9])
                print('sucessfully stored')

def compare_frame_size(path1, path2):
    filelist_valid_vid = os.listdir(path2)
    for vid in filelist_valid_vid:
        filelist_exp = os.listdir(os.path.join(path2, vid))
        for exp in filelist_exp:
            filelist_frame = os.listdir(os.path.join(path2, vid, exp))
            # print('video: {}, exp: {}'.format(vid, exp))
            for mask_name in filelist_frame:
                mask_path = os.path.join(path2, vid, exp, mask_name)
                mask_sample_path = os.path.join(path1, vid, exp, mask_name)
                mask_valid = Image.open(mask_path) 
                mask_sample = Image.open(mask_sample_path)
                valid_size = mask_valid.size
                sample_size = mask_sample.size
                bit_depth_valid = mask_valid.getbands()
                bit_depth_sample = mask_sample.getbands()
                if bit_depth_sample != bit_depth_valid:
                    print('wrong bit depth is {}_{}_{}'.format(vid, exp, mask_name))
                # else:
                #     print('{}_{}_{} bit depth no problems'.format(vid, exp, mask_name)) 
                    
                if valid_size != sample_size:
                    print('wrong id is {}_{}_{}'.format(vid, exp, mask_name))
                # else:
                #     print('{}_{}_{} size no problems'.format(vid, exp, mask_name))               

def get_img_info(path):
    # start_time = time.time()
    filelist_valid_vid = os.listdir(path)
    for vid in filelist_valid_vid:
        filelist_exp = os.listdir(os.path.join(path, vid))
        for exp in filelist_exp:
            filelist_frame = os.listdir(os.path.join(path, vid, exp))
            # print(filelist_frame)
            for mask_name in filelist_frame:
                mask_path = os.path.join(path, vid, exp, mask_name)
                mask = Image.open(mask_path)
                # contents = exifread.process_file(mask)
                print(mask.mode)
                
                
def pngquant_resize(path):
    start_time = time.time()
    filelist_valid_vid = os.listdir(path)
    for vid in filelist_valid_vid:
        filelist_exp = os.listdir(os.path.join(path, vid))
        for exp in filelist_exp:
            filelist_frame = os.listdir(os.path.join(path, vid, exp))
            # print(filelist_frame)
            for mask_name in filelist_frame:
                mask_path = os.path.join(path, vid, exp, mask_name)
                os.system("pngquant --quality=0-30 --speed=11 "+mask_path+" --force -o "+mask_path)
                print('sucessfully stored')
    end_time = time.time()
    print("用时：",round(end_time - start_time,2),"s")

def check_pseudo_number(path):
    json_pseudo = None
    with open(path, 'r') as f:
        json_pseudo = json.load(f)
    exp_ids = list(json_pseudo.keys())
    for exp_id in exp_ids:
        pseudo_labels = list(json_pseudo[exp_id])
        if len(pseudo_labels) < 3:
            print('wrong exp_id is {}'.format(exp_id))
    
if __name__ == "__main__" :
    # path1 = os.path.join(meta_path,"meta_expressions")
    
    # split_results(result_path)
    
    # check_file_number(valid_JPEG_path)
    # check_all_file_number(result_path)
    # compare_to_sample(sample_path,valid_path)
    
    pngquant_resize(valid_path)
    reduce_quality(valid_path)
    
    # compare_frame_size(sample_path,valid_path)
    # get_img_info(valid_path)