import json
import os
import argparse



def build_inference_testingset():
    json_path = 'COMPLEX_INSTRUCTIONS/ComplexBench/data/data_release.json'
    save_json_root = "COMPLEX_INSTRUCTIONS/ComplexBench/data_infer"

    save_json_path = os.path.join(save_json_root, os.path.basename(json_path).replace(".json", ".jsonl"))

    with open(json_path, "r") as fr:
        info_list = json.load(fr)


    with open(save_json_path, "w") as fw:
        for info in info_list:
            info["q"] = info["instruction"]
            fw.write(json.dumps(info, ensure_ascii=False)+"\n")
    return



def reformat_inference_results(json_path, save_json_path):
    # save_root = "COMPLEX_INSTRUCTIONS/ComplexBench/llm_generations_vllm_processed"
    # save_json_path = os.path.join(save_root, os.path.basename(json_path))
    assert(json_path != save_json_path)
    model_id = (os.path.basename(json_path)).split("_raw")[0]
    with open(save_json_path, "w") as fw:
        with open(json_path, "r") as fr:
            for line in fr:
                info = json.loads(line)
                info["model"] = model_id
                assert(model_id in info)
                info["generated"] = info.pop(model_id)
                fw.write(json.dumps(info, ensure_ascii=False)+"\n")

    return




if __name__ == "__main__":
    print()
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path_local", type=str, default="")
    parser.add_argument("--save_path", type=str, default="")

    args = parser.parse_args()

    json_path = args.input_path_local
    assert(os.path.exists(json_path))
    save_json_path = args.save_path
    os.makedirs(os.path.dirname(save_json_path), exist_ok=True)
    
    reformat_inference_results(json_path, save_json_path)

    
