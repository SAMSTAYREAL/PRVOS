import json
PSEUDO_LABEL_PATH = ''
##### put dictionary of pseudo labels into json file #####
def save_pseudo_lables(ref_id, labels, path):
    PSEUDO_LABEL_PATH = path
    vid = ref_id[0].split('_')[0]  
    oid = ref_id[0].split('_')[1]
    with open(path, 'r') as f:
        json_1 = json.load(f)
    ids = list(json_1["videos"].keys())
    for id in ids:
        if vid == id:
            exps = list(json_1["videos"][id]["expressions"].keys())
            for exp in exps:
                if oid == exp:
                    json_1["videos"][id]["expressions"][exp] = labels
    with open(path,"w") as json_new:
        json.dump(json_1,json_new)
######## for refer_datasets.py to get path #########    
def get_path():
    return PSEUDO_LABEL_PATH
    
##### load pseudo labels #####
def load_pseudo_labels(vid,oid,path):
    with open(path,'r') as f_labels:
        json_1 = json.load(f_labels)
    
    use_frame_index = json_1["videos"][vid]["expressions"][oid]
    return use_frame_index
              
        