import json

def make_json(file_1:str, epoch_num):
    json_1 = None
    json_2 = None
    with open(file_1, 'r') as f_obj1:
        json_1 = json.load(f_obj1)
    
    ids_1 = list(json_1["videos"].keys())
    
    for id_1 in ids_1:
        exps_1 = list(json_1["videos"][id_1]["expressions"].keys())
        for exp in exps_1:
            json_1["videos"][id_1]["expressions"][exp] = {}
    
    with open("/root/autodl-tmp/code/Refer-Youtube-VOS/Baseline/pseudo_labels_{}".format(epoch_num), "w") as json_new:
        json.dumps(json_1)
        
    path = "/root/autodl-tmp/code/Refer-Youtube-VOS/Baseline/pseudo_labels_{}".format(epoch_num)
    
    return path
        