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
        exp_2 = json_2["videos"][id_1]["expressions"]
        for exp_id in exp_2.keys():
            obj_id = exp_2[exp_id]["obj_id"]
            exp_to_add = []
            i = 0
            if obj_id in objs_1:
                while(i < exp_id*2):
                    exp_to_add.append(exp_2[i]['exp'])
                    exp_to_add.append(exp_2[i+1]['exp'])
                    json_1["videos"][id_1]["objects"][obj_id]["expressions"] = exp_to_add
                    exp_to_add.clear
                    i+2
                # exp_to_add = exp_2[exp_id]["exp"]
                # json_1["videos"][id_1]["objects"][obj_id]["expressions"] = exp_2[exp_id]["exp"]
                continue

    with open("./new.json" ,"w") as json_new1:
        json.dump(json_1, json_new1)

if __name__ == "__main__" :
    path_1 = "/home/imi005/datasets2/datasets/refer-yv-2019/Youtube-VOS/train/meta.json"
    path_2 = "/home//imi005/datasets2/datasets/refer-yv-2019/Youtube-VOS/train/meta_expressions.json"
    merge_json(path_1, path_2)    