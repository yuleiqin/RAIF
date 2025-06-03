import json
import os
import argparse
from glob import glob


def extract_jsonl_for_inference():
    jsonl_path = "COMPLEX_INSTRUCTIONS/CFBench/data/cfbench_data.json"
    save_jsonl_path = jsonl_path.replace(".json", "_infer.jsonl")
    with open(jsonl_path, "r") as fr:
        info_list = json.load(fr)

    assert(save_jsonl_path != jsonl_path)
    with open(save_jsonl_path, "w") as fw:
        for info in info_list:
            info["q"] = info["prompt"]
            fw.write(json.dumps(info, ensure_ascii=False)+"\n")

    return





def convert_inference2evaluation(input_path, output_path, model_id):
    info_list = []
    with open(input_path, "r") as fr:
        for line in fr:
            info = json.loads(line)
            assert(model_id in info)
            info["response"] = info.pop(model_id)
            info["infer_model"] = model_id
            info_list.append(info)
    
    with open(output_path, "w") as fw:
        json.dump(info_list, fw, ensure_ascii=False, indent=4)

    return




if __name__ == "__main__":
    print()
    ### convert from json to jsonlines
    # extract_jsonl_for_inference()

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="")
    parser.add_argument("--output_path", type=str, default="")
    parser.add_argument("--model_id", type=str, default="")

    args = parser.parse_args()
    
    input_path = args.input_path
    output_path = args.output_path
    model_id = args.model_id

    print("read-in json path", input_path)
    print("write-out path", output_path)
    print("model_id", model_id)
    assert(model_id != "")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    assert(os.path.isfile(input_path))

    convert_inference2evaluation(input_path, output_path, model_id)
