import json

def merge_json(file_1: str, file_2: str = ""):
    json_1 = None
    json_2 = None
    with open(file_1, 'r') as f_obj1:
        json_1 = json.load(f_obj1)
    with open(file_2, 'r') as f_obj2:
        json_2 = json.load(f_obj2)

    ids_1 = list(json_1["videos"].keys())
    
    for id_1 in ids_1:
        objs_1 = list(json_1["videos"][id_1]["objects"].keys())
        frame_list = json_2["videos"][id_1]["frames"]
        for obj_id in objs_1:
            json_1["videos"][id_1]["objects"][obj_id]["expressions"] = []
            json_1["videos"][id_1]["objects"][obj_id]["frames"] = frame_list
            
        
    for id_1 in ids_1:
        objs_1 = list(json_1["videos"][id_1]["objects"].keys())
        exp_2 = json_2["videos"][id_1]["expressions"]
        print('merging {}'.format(id_1))
        # i = int(list(exp_2.keys())[0])
        i=0
        print(i)
        
        for exp_id in exp_2.keys():
            obj_id = exp_2[exp_id]["obj_id"]
            if obj_id in objs_1:
                json_1["videos"][id_1]["objects"][obj_id]["expressions"].append(exp_2[exp_id]["exp"])
                
                
    with open("/home/imi1214/MJP/datasets/RVOS/Ref-DAVIS/train/new1.json" ,"w") as json_new:
        json.dump(json_1, json_new)

if __name__ == "__main__" :
    path_1 = "/home/imi1214/MJP/datasets/RVOS/Ref-DAVIS/train/meta.json"
    path_2 = "/home/imi1214/MJP/datasets/RVOS/Ref-DAVIS/train/meta_expressions.json"
    merge_json(path_1, path_2)    