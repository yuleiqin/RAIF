import os
import json
from glob import glob
from tqdm import tqdm
import numpy as np
import sys



def recursively_check(root_dir):
    json_pathlist = []
    visit = [root_dir]
    while len(visit):
        cur_path = visit.pop()
        json_pathlist += glob(os.path.join(cur_path, "*.json")) + glob(os.path.join(cur_path, "*.jsonl"))
        # json_pathlist += glob(os.path.join(cur_path, "input_data_ans_*.json"))
        if os.path.isdir(cur_path):
            subdirs = os.listdir(cur_path)
            for subdir in subdirs:
                visit.append(os.path.join(cur_path, subdir))
    json_pathlist = [item for item in json_pathlist if not (item.endswith("invalid.json"))]
    return json_pathlist



def convert(json_pathlist):
    for json_path in json_pathlist:
        visit_json_path = json_path + ".done"
        assert(visit_json_path != json_path)
        if os.path.exists(visit_json_path):
            print("Already visit JSON DIR, skip {}".format(json_path))
            continue
        print("Processing...{}".format(json_path))
        # try:
        res = {}
        with open(json_path, "r") as fr:
            # info_list = json.load(fr)
            lines_list = fr.readlines()

        for line in lines_list:
            info = json.loads(line)
            q = info.pop("prompt")
            if "key" in info:
                info.pop("key")
            if "instruction_id_list" in info:
                info.pop("instruction_id_list")
            if "kwargs" in info:
                info.pop("kwargs")
            for k,v in info.items():
                if not (k in res):
                    res[k] = []
                res[k].append({"prompt":q, "response":v})

        for k,v in res.items():
            save_json_path = os.path.join(os.path.dirname(json_path), "input_data_ans_{}.jsonl".format(k))
            assert(json_path!=save_json_path)
            # if os.path.exists(save_json_path):
            #     # print("Already exist JSON, skip {}".format(save_json_path))
            #     continue

            with open(save_json_path, "w") as fw:
                for line in v:
                    fw.write(json.dumps(line, ensure_ascii=False)+"\n")

        with open(visit_json_path, "w") as fw:
            fw.write("done")

        # except:
        #     print("Errors encountered for {}".format(json_path))
        #     continue

    return
                        



if __name__ == "__main__":

    if(len(sys.argv) == 1 or sys.argv[1] == "help"):
        usage()

    elif len(sys.argv) >= 2:
        root_dir = sys.argv[1]
        # json_pathlist = recursively_check(root_dir)
        json_pathlist = [root_dir]
        # print("json list", json_pathlist)
        convert(json_pathlist)
