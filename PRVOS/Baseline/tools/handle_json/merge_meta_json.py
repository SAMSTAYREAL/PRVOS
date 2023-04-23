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
        for obj_id in objs_1:
            json_1["videos"][id_1]["objects"][obj_id]["expressions"] = []
        
    for id_1 in ids_1:
        objs_1 = list(json_1["videos"][id_1]["objects"].keys())
        # exp_new = json_3["videos"][id_1]["objects"]
        exp_2 = json_2["videos"][id_1]["expressions"]
        print('merging {}'.format(id_1))
        # i = int(list(exp_2.keys())[0])
        for exp_id in exp_2.keys():
            exp2 = exp_2[exp_id]["exp"]
            obj_id = exp_2[exp_id]["obj_id"]
            if obj_id in objs_1:
                json_1["videos"][id_1]["objects"][obj_id]["expressions"].append(exp_2[exp_id]["exp"])
                # json_1["videos"][id_1]["objects"][obj_id]["exp_id"] = str(exp_id)
                # for i in range(len(json_1["videos"][id_1]["objects"][obj_id]["expressions"])):
                #     json_1["videos"][id_1]["objects"][obj_id]["expressions"]["exp_{}".format(i)]["exp"] = exp_2[exp_id]["exp"]
                #     json_1["videos"][id_1]["objects"][obj_id]["expressions"]["exp_{}".format(i)]["exp_id"] = str(exp_id)
                
                
    with open("/home/imi1214/MJP/datasets/RVOS/refer-yv-2019/Youtube-VOS/valid/new1.json" ,"w") as json_new:
        json.dump(json_1, json_new)


def Modiify_new1(file_3):
    json_3 = None
    with open(file_3, 'r') as f_obj3:
        json_3 = json.load(f_obj3)
    
    ids = list(json_3["videos"].keys())
    
    for id in ids:
        objs = list(json_3["videos"][id]["objects"].keys())
        for obj in objs:
            exps = json_3["videos"][id]["objects"][obj]["expressions"]
            print(exps)
            exp_num = len(exps)
            print(type(exp_num))
            for i in range(exp_num):
                json_3["videos"][id]["objects"][obj]["expressions"]["exp_{}".format(i)]["exp"] = exps[i]
                json_3["videos"][id]["objects"][obj]["expressions"]["exp_{}".format(i)]["exp_id"] = ""
    
    with open("/home/imi005/datasets2/datasets/refer-yv-2019/Youtube-VOS/valid/new.json", "w") as json_new:
        json.dump(json_3, json_new)
        

if __name__ == "__main__" :
    path_1 = "/home/imi1214/MJP/datasets/RVOS/refer-yv-2019/Youtube-VOS/valid/meta.json"
    path_2 = "/home/imi1214/MJP/datasets/RVOS/refer-yv-2019/Youtube-VOS/meta_expressions_test/meta_expressions/valid/meta_expressions.json"
    path_3 = "/home/imi005/datasets2/datasets/refer-yv-2019/Youtube-VOS/valid/new1.json"
    merge_json(path_1, path_2)  
    # Modiify_new1(path_3)  