import json
import yaml
import argparse
import os
import re
import concurrent.futures
import copy
from tqdm import tqdm
from run_local_model import call_model_service_infer
from utils import (
    load_data,
    chat_completion_openai,
    chat_completion_openai_azure,
    chat_completion_anthropic,
    load_cache_data,
    get_endpoint,
    make_config,
    reorg_file
)
import json5
import yaml
from openai import OpenAI
from transformers import AutoTokenizer
from copy import deepcopy


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
    inputs = tokenizer(response, return_tensors="pt")
    input_ids = inputs["input_ids"][0]
    return len(input_ids)


def count_token_max(response, tokenizer_list):
    response_token_len_list = [count_token(response, tokenizer_item) for tokenizer_item in tokenizer_list]
    return max(response_token_len_list)


global tokenizer
tokenizer = build_model()
global tokenizer_mistral
tokenizer_mistral = build_model_mistral()
global tokenizer_llama
tokenizer_llama = build_model_llama()



global client

def get_answer(model, conv, temperature, max_tokens, endpoint_dict=None, require_json=False):
    '''get answer from model, for judge model (e.g. gpt4)'''
    api_dict = get_endpoint(endpoint_dict["endpoints"])

    if endpoint_dict["api_type"] == "anthropic":
        output = chat_completion_anthropic(model, conv, temperature, max_tokens)
    elif endpoint_dict["api_type"] == "azure":
        output = chat_completion_openai_azure(model, conv, temperature, max_tokens, api_dict)
    else:
        output = chat_completion_openai(model, conv, temperature, max_tokens, api_dict, require_json=require_json)
    return output



def check_gpt_judge(gpt_judge, language):
    '''The input is a dictionary (where the key is "checkpoint", and the value is another dictionary containing "judgement reason", "judgement result", and "weight"), and the output is also a dictionary.'''
    for ckpt, judge in gpt_judge.items():
        assert isinstance(ckpt, str) and ckpt, f"checkpoint {ckpt} should be a non-empty string, but got {ckpt}"
        assert isinstance(judge, dict), f"checkpoint {ckpt} judge should be a non-empty dict, but got {judge}"
        
        if language == 'en':
            assert "judgement reason" in judge and "judgement result" in judge and "weight" in judge, f"checkpoint {ckpt} judge should have keys 'judgement reason', 'judgement result' and 'weight', but got {judge}"
            assert isinstance(judge["judgement reason"], str) and judge["judgement reason"], f"checkpoint {ckpt} judge 'judgement reason' should be a non-empty string, but got {judge['judgement reason']}"
            assert judge["judgement result"] in ["yes", "no"], f"checkpoint {ckpt} judge 'judgement result' should be 'yes' or 'no', but got {judge['judgement result']}"
        elif language == 'zh':
            assert "评判理由" in judge and "评判结果" in judge and "weight" in judge, f"checkpoint {ckpt} judge should have keys '评判理由', '评判结果' and 'weight', but got {judge}"
            assert isinstance(judge["评判理由"], str) and judge["评判理由"], f"checkpoint {ckpt} judge '评判理由' should be a non-empty string, but got {judge['评判理由']}"
            assert judge["评判结果"] in ['是', '否'], f"checkpoint {ckpt} judge '评判结果' should be '是' or '否', but got {judge['评判结果']}"
        
        assert judge["weight"] is None or ( (isinstance(judge["weight"], int) or (isinstance(judge["weight"], float)) ) and 0<judge["weight"]<=1 ), f"checkpoint {ckpt} judge 'weight' should be None or an integer between 0 and 1, but got {judge['weight']}"
    return gpt_judge
        

def split_checklist(checklist):
    gpt_checklist = {}
    heuristic_checklist = {}
    
    gpt_checklist = checklist
    
    return gpt_checklist, heuristic_checklist



def heuristic_judgement(question, heuristic_checklist):
    return None


def get_score(final_judge, language):
    '''"judge" is a dictionary, where the key is "checkpoint", and the value is another dictionary containing "judgement reason", "judgement result", and "weight"'''
    score = 0
    for ckpt, judge in final_judge.items():
        if language == 'en':
            if judge["judgement result"] == "yes":
                if judge["weight"]:
                    score += judge["weight"]
                else:
                    return 1
        elif language == 'zh':
            if judge["评判结果"] == "是":
                if judge["weight"]:
                    score += judge["weight"]
                else:
                    return 1
        else:
            raise ValueError(f"language should be 'en' or 'zh', but got {language}")
    return score



def post_process_json(json_str):
    json_raw = copy.deepcopy(json_str)
    if "```json\n" in json_str:
        json_str = json_str[json_str.find("```json\n")+len("```json\n"):json_str.rfind("\n```")]

    elif "```\n" in json_str:
        json_str = json_str[json_str.find("```\n")+len("```\n"):json_str.rfind("\n```")]
    else:
        pass   # 不做任何改变
        
    while "\n\n" in json_str:
        json_str = json_str.replace("\n\n", "\n")
    
    # import pdb;pdb.set_trace();
    json_dict = None
    is_invalid = False
    try:
        json_dict = json5.loads(json_str)
    except:
        try: 
            json_dict = yaml.safe_load(json_str)
        except:
            try:
                json_dict = eval(json_str)
            except: 
                is_invalid = True
    
    # import pdb;pdb.set_trace();
    if is_invalid:
        return None
    
    else:
        return json_dict



def judgment(**args):
    cur_try = 0
    global client
    global tokenizer

    while cur_try < 3:
        cur_try += 1
        try:
            cur_question = args["question"]
            configs = args["configs"]
            output_file = args["output_file"]
            # model = configs["judge_model"]
            model = args["judge_model"]

            # uniform the checklist format
            cur_checklist = cur_question['checklist']
            if isinstance(cur_checklist[0], str):
                cur_checklist = [[ckpt, None] for ckpt in cur_checklist]

            gpt_checklist, heuristic_checklist = split_checklist(cur_checklist)
            
            gpt_checklist_judgement = {}
            for ckpt, weight in gpt_checklist:
                # gpt_checklist_judgement[ckpt] = {'judgement reason': "", "judgement result": "", "weight": weight}
                gpt_checklist_judgement[ckpt] = {'评判理由': "", "评判结果": "", "weight": weight}

            conv = []
            # if 'system_prompt' in configs and configs['system_prompt']:
            #     conv = [{"role": "system", "content": configs["system_prompt"]}]
            conv.append({"role": "system", "content": "You are a helpful assistant."})

            if configs["prompt_language"]=='zh':
                prompt_template = configs["prompt_template_zh"]
            elif configs["prompt_language"]=='en':
                prompt_template = configs["prompt_template_en"]
            else:
                raise ValueError(f"prompt_language should be 'zh' or 'en', but got {configs['prompt_language']}")
            
            eval_prompt = prompt_template.format(
                        user_query=cur_question['user_query'],
                        origin_first_response=cur_question['origin_first_response'],
                        feedback=cur_question['feedback'],
                        second_response=cur_question['second_response'],
                        reference_second_response=cur_question['reference_second_response'],
                        checklist=cur_question['checklist'],
                        checklist_judgement=json.dumps(gpt_checklist_judgement, ensure_ascii=False, indent=4),
                    )

            # print(eval_prompt)
            conv.append({"role": "user", "content": eval_prompt})
            
            messages_str = ""
            for message in conv:
                messages_str += message["content"]+"\n"
            max_tokens = max(int((16384 - count_token_max(messages_str, [tokenizer, tokenizer_llama, tokenizer_mistral]))// 1.1 - 100), 1)
            # response = call_model_service_infer(conv, model)
            response = client.chat.completions.create(
                                        model=model,  
                                        messages=conv,
                                        max_tokens=max_tokens,
                                        temperature=0).to_dict()
            assert "choices" in response
            response = response["choices"][0]["message"]["content"]

            # print(">>> RAW response", response)
            if response == '$ERROR$':
                print("API failed, output is ERROR!")
                continue

            if response == '$REPETITIVE PATTERNS$':
                print("detect repetitive patterns")
                final_judge = {'API fialed': "$REPETITIVE PATTERNS$"}
                score = 0
                continue

            response_json = post_process_json(response)
            # import pdb;pdb.set_trace();
            if response_json is None:
                print("JSON INVALID! output cannot be read into JSON dictionary")
                continue
            
            else:
                response = json.dumps(response_json, ensure_ascii=False)
                # print(">>> PROCESSED response", response)

            gpt_judge = check_gpt_judge(response_json, configs['prompt_language'])
            # import pdb;pdb.set_trace();

            if heuristic_checklist:
                heuristic_judge = heuristic_judgement(cur_question, heuristic_checklist)
            else:
                # print("There is no heuristic checklist")
                heuristic_judge = {}
            
            # import pdb;pdb.set_trace();
            final_judge = {**gpt_judge, **heuristic_judge}
            # import pdb;pdb.set_trace();

            score = get_score(final_judge, configs['prompt_language'])
            # import pdb;pdb.set_trace();

            assert 0-1e-6<=score<=1+1e-6, f"score should be between 0 and 1, but got {score}, corrsponding to final_judge {final_judge}, corresponding to user_query {cur_question['user_query'][:10]}"

            output = {
                **cur_question,
                "judge_model": model,
                "eval_prompt": eval_prompt,
                "judgement": final_judge,
                "score": score 
                }
            
            # if "infer_model" in cur_question:
            #     output["infer_model_in_judgepy"] = cur_question['infer_model']
            
            # if "second_response" in cur_question:
            #     output["second_resp_in_judgepy"] = cur_question['second_response'],

            with open(output_file, "a+") as f:
                f.write(json.dumps(output, ensure_ascii=False) + "\n")
                f.flush()
            break
            
        except Exception as e:
            print(f"Error: {e}")
            if cur_try < 3:
                print(f"Retry {cur_try} times")
            else:
                print(f"Failed after {cur_try} times")
                return None
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting-file", type=str, default="config/judge_config.yaml")
    parser.add_argument("--endpoint-file", type=str, default="config/api_config.yaml")
    parser.add_argument("--out_dir", type=str, default="evaluation")
    parser.add_argument("--judge_out_dir", type=str, default="judgement")
    parser.add_argument("--batch_size", type=str, default=16)
    parser.add_argument(
        "--model_id", type=str, default=""
    )
    parser.add_argument(
        "--judge_model_id", type=str, default=""
    )
    parser.add_argument(
        "--judge_model_url", type=str, default=""
    )
    args = parser.parse_args()
    print(args)

    configs = make_config(args.setting_file)
    endpoint_list = make_config(args.endpoint_file)

    print(json.dumps(configs, indent=4, ensure_ascii=False))
    # print(f'judge model: {configs["judge_model"]} \ntemperature: {configs["temperature"]}, max tokens: {configs["max_tokens"]}')
    # print("System prompt: ", configs["system_prompt"] if configs["system_prompt"] else "None")
    # print("Prompt template: ")
    # print(configs["prompt_template"])

    question_file = os.path.join("data", configs["bench_name"], configs["test_file_name"])
    # answer_dir = os.path.join("data", configs["bench_name"], "model_answer")
    answer_dir = args.out_dir
    assert(os.path.exists(answer_dir))
    judge_model_id = args.judge_model_id
    judge_model_url = args.judge_model_url
    global client
    print("加载模型服务地址：", judge_model_url)
    client = OpenAI(base_url=judge_model_url, api_key = "xxx", timeout=600)
    print("加载模型名称：", judge_model_id)

    questions = load_data(question_file)
    model_answers = load_cache_data(answer_dir)
    # Organize the data in "model_answers" into a dictionary using "user_query" as the key.
    model_answers_dict = {model: {data_dict['user_query']: data_dict for data_dict in data} for model, data in model_answers.items()}
    
    # if user choose a set of models, only judge those models
    # models = [model for model in configs["model_list"]]
    model = args.model_id
    models = [model]
    output_files = {}
    # output_dir = f"data/{configs['bench_name']}/model_judgment/{configs['judge_model']}"
    output_dir = args.judge_out_dir
    os.makedirs(output_dir, exist_ok=True)
    for model in models:
        output_files[model] = os.path.join(
            output_dir,
            f"{model}_judged_by_{judge_model_id}.jsonl",
        )

    for output_file_path in output_files.values():
        print("Judgement Save file to", output_file_path)
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    existing_judgments = load_cache_data(output_dir)
    print("len(existing_judgments)", len(existing_judgments))
    # import pdb;pdb.set_trace();
    model_visited = f"{model}_judged_by_{judge_model_id}"
    # import pdb;pdb.set_trace();
    if model_visited in existing_judgments:
        print(f"len(existing_judgments[{model_visited}])", len(existing_judgments[model_visited]))

    # endpoint_info = endpoint_list[configs["judge_model"]]

    # parallel = endpoint_info["parallel"]
    parallel = int(args.batch_size)
    print("Send Evaluation by Parallel", parallel)

    with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as executor:
        futures = []
        for model in models:
            print("number of candidate judgments", len(questions))
            count = 0
            for question in questions:
                
                kwargs = {}
                kwargs["question"] = copy.deepcopy(question)

                if model in model_answers and model_answers[model] and not question['user_query'] in set(data_dict['user_query'] for data_dict in model_answers[model]):
                    print(f"Warning: {model} answer to 【{question['user_query']}】 cannot be found.")
                    continue

                if model_visited in existing_judgments and existing_judgments[model_visited] and question['user_query'] in set(data_dict['user_query'] for data_dict in existing_judgments[model_visited]):
                    count += 1
                    continue
                
                # import pdb;pdb.set_trace();
                # Based on the model's name "h" and "user_query", find the corresponding "second_response" and "infer_model".
                if model in model_answers_dict and question['user_query'] in model_answers_dict[model]:
                    kwargs['question']['second_response'] = model_answers_dict[model][question['user_query']]['second_response']
                    kwargs['question']['infer_model'] = model_answers_dict[model][question['user_query']]['infer_model']
                    assert model == kwargs['question']['infer_model'], f"model name in model_answers_dict should be the same as model name in configs, but got {model} and {kwargs['question']['infer_model']}"
                else:
                    print(f"Warning: {model} answer to 【{question['user_query']}】 cannot be found.")
                    continue

                kwargs["configs"] = configs
                # kwargs["endpoint_dict"] = endpoint_info
                kwargs["output_file"] = output_files[model]
                kwargs["judge_model"] = judge_model_id
                future = executor.submit(judgment, **kwargs)
                futures.append(future)

            if count > 0:
                print(f"{model} {count} number of existing judgments")

        if len(futures):
            for future in tqdm(
                concurrent.futures.as_completed(futures), total=len(futures)
            ):
                future.result()

    # Reorder the judgement results of the model.
    for model in models:
        output_file = output_files[model]
        reorg_file(output_file, sort_key="user_query")
        