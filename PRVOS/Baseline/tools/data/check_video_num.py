###########  check the number of the videos in meta_expressions.json  ###########
import json
import pickle

def check_video_num(file_name:str):
    json_1 = None
    
    with open(file_name, 'r') as f:
        json_1 = json.load(f)
        
    ids_1 = list(json_1["videos"].keys())
    
    print(len(ids_1))
    
def check_frames_num(file_1:str, file_2:str):
    json_1 = None
    json_2 = None

def test_txt_input():
    pseudo = {
        'a':1,
        'b':2
    }
    with open('pseudo_lable.txt','wb+') as file:
        file.write(pickle.dumps(pseudo))
    with open('pseudo_label.json','w') as f:
        json.dump(pseudo,f)
    

if __name__ == "__main__" :
    # path = "/root/autodl-tmp/datasets/refer-yv-2019/Youtube-VOS/meta_expressions_test/test/meta_expressions.json"
    # check_video_num(path)
    test_txt_input()