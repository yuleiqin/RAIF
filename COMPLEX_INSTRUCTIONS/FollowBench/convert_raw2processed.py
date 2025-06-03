import os
import json
import argparse
from glob import glob



def convert_from_raw2processed(input_path, output_path, model_id, judge_model_id):
    jsonl_pathlist = glob(os.path.join(input_path, "*.jsonl"))
    for jsonl_path in jsonl_pathlist:
        if model_id == "":
            save_jsonl_path = os.path.join(output_path, os.path.basename(jsonl_path))
        else:
            save_jsonl_path = os.path.join(output_path, str(model_id) + "_" + os.path.basename(jsonl_path))

        with open(jsonl_path, "r") as fr:
            with open(save_jsonl_path, "w") as fw:
                for line in fr:
                    info = json.loads(line)
                    if model_id != "":
                        assert(model_id in info)
                        outputs = info.pop(model_id)
                    elif judge_model_id != "":
                        assert(judge_model_id in info)
                        outputs = info.pop(judge_model_id)
                    else:
                        raise ValueError("Model-id or Judge Model-id MUST be valid")
                        
                    info['choices'][0]['message']['content'] = outputs
                    fw.write(json.dumps(info, ensure_ascii=False)+"\n")

    return



if __name__ == "__main__":
    print()
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path_local", type=str, default="")
    parser.add_argument("--output_path", type=str, default="")
    parser.add_argument("--model_id", type=str, default="")
    parser.add_argument("--judge_model_id", type=str, default="")
    args = parser.parse_args()
    
    input_path_local = args.input_path_local
    output_path = args.output_path
    model_id = args.model_id
    judge_model_id = args.judge_model_id

    print("read-in json path", input_path_local)
    print("write-out path", output_path)
    assert(os.path.exists(input_path_local))

    convert_from_raw2processed(input_path_local, output_path, model_id, judge_model_id)



