import json
import os
import argparse
from glob import glob



def combine_jsonl_by_filename(input_path, output_path):
    input_jsonpathlist = glob(os.path.join(input_path, "*.jsonl"))
    with open(output_path, "w") as fw:
        for input_jsonpath in input_jsonpathlist:
            input_filename = os.path.basename(input_jsonpath)
            with open(input_jsonpath, "r") as fr:
                for line in fr:
                    info = json.loads(line)
                    info["source"] = input_filename
                    fw.write(json.dumps(info, ensure_ascii=False)+"\n")

    return




def split_jsonl_by_filename(input_path, output_path, model_id):
    input_by_source = {}
    with open(input_path, "r") as fr:
        for line in fr:
            info = json.loads(line)
            source = info["source"]
            response = info.pop(model_id)
            if type(response) is dict and "content" in response:
                response = response["content"]
            info["model_answer"] = response

            if not (source in input_by_source):
                input_by_source[source] = []
            input_by_source[source].append(info)

    for source in input_by_source:
        output_json_path = os.path.join(output_path, source.replace(".jsonl", ".json"))
        with open(output_json_path, "w") as fw:
            # for line in input_by_source[source]:
            #     fw.write(line)
            json.dump(input_by_source[source], fw, ensure_ascii=False)

    return





if __name__ == "__main__":
    print()
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="")
    parser.add_argument("--output_path", type=str, default="")
    parser.add_argument("--is_combine", type=str, default="")
    parser.add_argument("--model_id", type=str, default="")

    args = parser.parse_args()
    
    input_path = args.input_path
    output_path = args.output_path
    is_combine = (args.is_combine == "True")
    model_id = args.model_id

    print("是否是合并所有jsonl文件为一个整体", is_combine)
    print("read-in json path", input_path)
    print("write-out path", output_path)
    print("model_id", model_id)

    assert(os.path.exists(input_path))
    if is_combine:
        assert(os.path.isdir(input_path))
        combine_jsonl_by_filename(input_path, output_path)
    else:
        assert(os.path.isfile(input_path))
        assert(os.path.isdir(output_path))
        os.makedirs(output_path, exist_ok=True)
        split_jsonl_by_filename(input_path, output_path, model_id)


    