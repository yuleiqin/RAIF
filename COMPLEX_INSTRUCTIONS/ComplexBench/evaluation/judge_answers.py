import os
import json
import argparse
import re



def save_jsonl(entry, sava_path):
    with open(sava_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(entry, ensure_ascii=False)+ "\n")
    return




# def extract_answers(local_json_input, output_path, model_id):
#     with open(local_json_input, "r") as fr:
#         for line in fr:
#             input_data = json.loads(line)
#             assert(model_id in input_data)
#             entry = input_data["data"]
#             entry['ass'] = input_data[model_id]
#             entry["payload"]["model"] = model_id
#             save_jsonl(entry, output_path)

#     return


def extract_judges(local_json_input, output_path, model_id):
    with open(output_path, "w", encoding='utf-8') as fw:
        with open(local_json_input, "r") as fr:
            for line in fr:
                input_data = json.loads(line)
                assert(model_id in input_data)
                entry = input_data["data"]
                generation = input_data[model_id]
                entry["payload"]["model"] = model_id

                re_result = re.findall(r'答案：是|答案：否', generation)
                if len(re_result) == 1:
                    if "是" in re_result[0]:
                        entry['point_judge'] = True
                    else:
                        entry['point_judge'] = False
                else: 
                    if "是" in generation and "否" not in generation:
                        entry['point_judge'] = True
                    else:
                        entry['point_judge'] = False
                
                entry['point_explanation'] = generation
                fw.write(json.dumps(entry, ensure_ascii=False)+ "\n")

    return




if __name__ == "__main__":
    print()
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path_local", type=str, default="")
    parser.add_argument("--output_path", type=str, default="")
    parser.add_argument("--model_id", type=str, default="")
    args = parser.parse_args()
    
    input_path_local = args.input_path_local
    output_path = args.output_path
    print("read-in json path", input_path_local)
    print("write-out path", output_path)
    assert(os.path.exists(input_path_local))
    model_id = args.model_id
    print("the response is judged by {}".format(model_id))

    extract_judges(input_path_local, output_path, model_id)











