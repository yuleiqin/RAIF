import json
import os
import time
import tiktoken
import argparse
import re
import concurrent.futures
from os.path import join,exists
from openai import OpenAI
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers import AutoTokenizer



encoder = tiktoken.get_encoding("cl100k_base")

SYS_MSG ="Based on the provided Input (if any) and Generated Text, answer the ensuing Questions with either a YES or NO choice. Your selection should be based on your judgment as well as the following rules:\n\n- YES: Select 'YES' if the generated text entirely fulfills the condition specified in the question. However, note that even minor inaccuracies exclude the text from receiving a 'YES' rating. As an illustration. consider a question that asks. \"Does each sentence in the generated text use a second person?” If even one sentence does not use the second person, the answer should NOT be 'YES'. To qualify for a 'YES' rating, the generated text must be entirely accurate and relevant to the question\n\n- NO: Opt for 'NO' if the generated text fails to meet the question's requirements or provides no information that could be utilized to answer the question. For instance, if the question asks. \"Is the second sentence in the generated text a compound sentence?\" and the generated text only has one sentence. it offers no relevant information to answer the question. Consequently, the answer should be 'NO'.'''"



def build_model():
    model_name_or_path = "/apdcephfs_cq8/share_2992827/shennong_5/ianxxu/pretrained_models/Qwen2.5-7B-Instruct_Qwen"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
    print("loaded tokenizer")
    return tokenizer


def build_model_mistral():
    model_name_or_path = "/apdcephfs_cq8/share_2992827/shennong_5/ianxxu/pretrained_models/Mistral-7B-Instruct-v0.3"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
    print("loaded tokenizer")
    return tokenizer


def build_model_llama():
    model_name_or_path = "/apdcephfs_cq8/share_2992827/shennong_5/ianxxu/pretrained_models/Meta-Llama-3.1-8B-Instruct_meta-llama"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
    print("loaded tokenizer")
    return tokenizer


def count_token(response, tokenizer):
    try:
        inputs = tokenizer.apply_chat_template(response, tokenize=True)
    except:
        inputs = tokenizer.encode(response)
    return len(inputs)


def count_token_max(response, tokenizer_list):
    response_token_len_list = [count_token(response, tokenizer_item) for tokenizer_item in tokenizer_list]
    return max(response_token_len_list)


global tokenizer
tokenizer = build_model()
global tokenizer_mistral
tokenizer_mistral = build_model_mistral()
global tokenizer_llama
tokenizer_llama = build_model_llama()


def load_jsonl(file_path):
    "General function to load jsonl file"
    _data = []
    with open(file_path, 'r') as f:
        for data in f:
            jline = json.loads(data)
            _data.append(jline)
    return _data


def bool_ratio(fpath):
    "Calculate true false ratio for eval results"
    _data = load_jsonl(fpath)
    count = {"true":0, "false":0}
    visited_ids = set()

    instruction_level_hard =  {"true":0, "false":0}
    instruction_level_easy =  {"true":0, "false":0}
    prompt_level_hard =  {"true":0, "false":0}
    prompt_level_easy =  {"true":0, "false":0}

    for entry in _data:
        if entry['id'] in visited_ids:
            continue

        if entry.get("eval", None) is None:
            print("Wrong output")
            print(entry['id'])

        if len(entry['decomposed_questions']) != len(entry['eval']):
            print("Wrong length")
            print(entry['id'])

        if None in entry['eval']:
            print("None in eval")
            print(entry['id'])
        
        num_instruction = len(entry['eval'])
        num_correct = 0
        for eva_value in entry['eval']:
            if eva_value:
                count["true"] += 1
                num_correct += 1
            else:
                count["false"] += 1
        if num_correct == num_instruction:
            prompt_correct = 1
        else:
            prompt_correct = 0
        
        if "easy" in (entry["subset"]).lower():
            instruction_level_easy["true"] += num_correct
            instruction_level_easy["false"] += (num_instruction-num_correct)
            prompt_level_easy["true"] += prompt_correct
            prompt_level_easy["false"] += (1-prompt_correct)

        else:
            instruction_level_hard["true"] += num_correct
            instruction_level_hard["false"] += (num_instruction-num_correct)
            prompt_level_hard["true"] += prompt_correct
            prompt_level_hard["false"] += (1-prompt_correct)

        visited_ids.add(entry['id'])
    
    print("-------- True False Table --------")
    print(count)
    print(f"Percentage of True: {count['true']/(sum(count.values())+1e-4)}")

    stats_jsonl_path = fpath + "_score.jsonl"
    res = {}
    res["overall"] = float(count['true']/(sum(count.values())+1e-4))
    res["count"] = count
    instruction_level_easy_true = instruction_level_easy["true"]/(sum(instruction_level_easy.values())+1e-4)
    prompt_level_easy_true = prompt_level_easy["true"]/(sum(prompt_level_easy.values())+1e-4)
    instruction_level_hard_true = instruction_level_hard["true"]/(sum(instruction_level_hard.values())+1e-4)
    prompt_level_hard_true = prompt_level_hard["true"]/(sum(prompt_level_hard.values())+1e-4)
    
    instruction_level_all_true = (instruction_level_easy["true"]+instruction_level_hard["true"])/(sum(instruction_level_easy.values())+sum(instruction_level_hard.values())+1e-4)
    prompt_level_all_true = (prompt_level_easy["true"]+prompt_level_hard["true"])/(sum(prompt_level_easy.values())+sum(prompt_level_hard.values())+1e-4)
    
    res["prompt_total_count"] = sum(prompt_level_hard.values()) + sum(prompt_level_easy.values())
    assert(res["prompt_total_count"]==500)
    res["instruction_level_easy_true"] = instruction_level_easy_true
    res["prompt_level_easy_true"] = prompt_level_easy_true
    res["instruction_level_hard_true"] = instruction_level_hard_true
    res["prompt_level_hard_true"] = prompt_level_hard_true
    res["instruction_level_all_true"] = instruction_level_all_true
    res["prompt_level_all_true"] = prompt_level_all_true

    with open(stats_jsonl_path, "w") as fw:
        fw.write(json.dumps(res, ensure_ascii=False)+"\n")
    
    return



def run_evaluation(client, in_path, o_dir, eval_model="gpt-4-0314", temperature=0, batch_size=4):
    """
    Main function to run decomposed questisons evaluation on models' outputs
        in_path: str, path to the model output file
        o_dir: str, path to the output folder
        eval_model: str, default "gpt-4-0314", model name to be used for evaluation
        temperature: float, default 0, temperature to be used for evaluation
    """
    _data = load_jsonl(in_path)
    # _model_name = in_path.split('/')[1].split('_')[0]
    _model_name = os.path.basename(os.path.dirname(in_path))
    print("evaluating {}".format(_model_name))
    # ceate output folder if not exists
    _o_dir = join(o_dir, eval_model)
    if not exists(_o_dir):
        os.mkdir(_o_dir)

    _opath = join(_o_dir, f"{_model_name}_DecomposeEval.json")
    # load_results if exists    
    result_writer = open(_opath, 'w')
    
    print(f"--------Evaluating output from {in_path}--------")
    print(f"--------Evaluation Using {eval_model}--------")
    parallel = int(batch_size)
    # print("Send Evaluation by Parallel", parallel)
    global tokenizer
    global tokenizer_llama
    global tokenizer_mistral

    for entry in tqdm(_data):
        message = []
        input_task = entry['input']
        output = entry[_model_name]
        num_output_token = count_token_max(output, [tokenizer, tokenizer_llama, tokenizer_mistral])
        # 保证输入token数目不要超
        while num_output_token > 12000:
            output = output[:-1000]
            num_output_token = count_token_max(output, [tokenizer, tokenizer_llama, tokenizer_mistral])

        # assert(len(output))
        # print(f"--------Instance {entry['id']}--------")
        for question_idx, question in enumerate(entry['decomposed_questions']):
            if len(message) == 0:
                if input_task:
                    content =  f"{SYS_MSG}\n\nInput:\n\"{input_task}\"\n\nGenerated Text:\n\"{output}\"\n\nQuestion:\n{question}\n"
                else:
                    content =  f"{SYS_MSG}\n\nGenerated Text:\n\"{output}\"\n\nQuestion:\n{question}\n"
            else:
                content = f"{question}\n"
            message.append({"role": "user", "content": content})
            # create a chat completion
            entry["question_idx"] = question_idx
            entry["conversations"] = message

            result_writer.write(json.dumps(entry, ensure_ascii=False) + '\n')
            result_writer.flush()
    
    return _opath



def run_evaluation_compute(output_jsonl_path, eval_model):
    """
    Main function to run decomposed questisons evaluation on models' outputs
        in_path: str, path to the model output file
        o_dir: str, path to the output folder
        eval_model: str, default "gpt-4-0314", model name to be used for evaluation
        temperature: float, default 0, temperature to be used for evaluation
    """
    output_jsonl_path_ans = output_jsonl_path.replace(".jsonl", "_final.jsonl")
    assert(output_jsonl_path_ans != output_jsonl_path)

    res = {}
    with open(output_jsonl_path, "r") as fr:
        for line in fr:
            info = json.loads(line)
            info_idx = info["id"]
            if not (info_idx in res):
                res[info_idx] = info
            if not ("question_res" in res[info_idx]):
                res[info_idx]["question_res"] = {}
            question_idx = info["question_idx"]
            res[info_idx]["question_res"][question_idx] = info[eval_model]

    with open(output_jsonl_path_ans, "w") as fw:
        for info_idx in res:
            info = res[info_idx]
            bool_results = []

            for question_idx in range(len(info["decomposed_questions"])):
                if question_idx in res[info_idx]["question_res"]:
                    res_question_idx = res[info_idx]["question_res"][question_idx]
                    res_question_idx = str(res_question_idx.strip()).lower()
                    if "yes" in res_question_idx:
                        bool_results.append(True)
                    elif "no" in res_question_idx:
                        bool_results.append(False)
                    else:
                        bool_results.append(None)
                else:
                    bool_results.append(None)
                    
            info['eval'] = bool_results
            fw.write(json.dumps(info, ensure_ascii=False)+"\n")
    
    bool_ratio(output_jsonl_path_ans)

    return



def main_run(args):

    client = OpenAI(base_url=args.model_url, api_key = "xxx", timeout=600)
    results_file = args.input
    output_dir = args.output_dir
    eval_model = args.model
    temperature = args.temperature
    batch_size = args.batch_size
    
    if not exists(results_file):
        print(f"results_dir {results_file} not exists")
        return
    
    if args.compute_metric == 0:
        # run evaluation for each model
        run_evaluation(client, results_file, output_dir, eval_model, temperature, batch_size) 
    else:
        # run evaluation for calculation
        run_evaluation_compute(results_file, eval_model)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--model", type=str, default="gpt-4-0314", help="model name to be used for evaluation")
    parser.add_argument("--model_url", type=str, default="localhost", help="model path")
    parser.add_argument("--input", type=str, required=True, help="path to the results file")
    parser.add_argument("--output_dir", type=str, required=True, help="path to the output folder")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--compute_metric", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0, help="temperature to be used for evaluation")
    args = parser.parse_args()
    main_run(args)
