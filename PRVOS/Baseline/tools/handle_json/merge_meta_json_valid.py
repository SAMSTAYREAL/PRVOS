import json

def merge_json(file_1: str, file_2:str, file_3: str = ""):
    json_1 = None
    json_2 = None
    with open(file_1, 'r') as f_obj1:
        json_1 = json.load(f_obj1)
    with open(file_2, 'r') as f_obj2:
        json_2 = json.load(f_obj2)
    with open(file_3, 'r') as f_obj3:
        json_3 = json.load(f_obj3)

    ids_1 = list(json_1["videos"].keys())
    
    for id_1 in ids_1:  # id_1 is the id of the video
        objs_1 = list(json_1["videos"][id_1]["objects"].keys())
        for obj_id in objs_1:
            json_1["videos"][id_1]["objects"][obj_id]["expressions"] = []
            json_1["videos"][id_1]["objects"][obj_id]["frames"] = []
        
    for id_1 in ids_1:
        objs_1 = list(json_1["videos"][id_1]["objects"].keys()) # objs_1 is the list of the object numbers
        exp_2 = json_2["videos"][id_1]["expressions"]
        exp_3 = json_3["videos"][id_1]["expressions"]
        fm_2 = list(json_2["videos"][id_1]["frames"])
        print('merging {}'.format(id_1))
        # i = int(list(exp_2.keys())[0])
        objs_1_1 = list(json_1["videos"][id_1]["objects"].keys())
        for obj_id_1 in objs_1_1:
            for i in range(len(fm_2)):
                json_1["videos"][id_1]["objects"][obj_id_1]["frames"].append(fm_2[i])
        for exp_id in exp_2.keys():
            obj_id_2 = exp_3[exp_id]["obj_id"]
            if obj_id_2 in objs_1:
                json_1["videos"][id_1]["objects"][obj_id_2]["expressions"].append(exp_2[exp_id]["exp"])
            # json_1["videos"][id_1]["objects"][obj_id]["expressions"].append(exp_2[exp_id]["exp"])
                
                
    with open("/home/imi005/datasets2/datasets/refer-yv-2019/Youtube-VOS/valid/new.json" ,"w") as json_new:
        json.dump(json_1, json_new)

if __name__ == "__main__" :
    path_1 = "/home/imi1214/MJP/datasets/RVOS/refer-yv-2019/Youtube-VOS/valid/meta.json"
    path_2 = "/home/imi1214/MJP/datasets/RVOS/refer-yv-2019/Youtube-VOS/meta_expressions_test/meta_expressions/valid/meta_expressions.json"
    path_3 = "/root/autodl-tmp/datasets/refer-yv-2019/Youtube-VOS/valid/meta_expressions.json"
    merge_json(path_1, path_2, path_3)    