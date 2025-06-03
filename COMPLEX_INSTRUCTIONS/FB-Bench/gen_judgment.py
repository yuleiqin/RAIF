import json
import yaml
import argparse
import os
import re
import concurrent.futures
import copy
from tqdm import tqdm

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

def judgment(**args):
    cur_try = 0
    while cur_try < 3:
        cur_try += 1
        try:
            cur_question = args["question"]
            configs = args["configs"]
            output_file = args["output_file"]
            model = configs["judge_model"]
            
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
            if 'system_prompt' in configs and configs['system_prompt']:
                conv = [{"role": "system", "content": configs["system_prompt"]}]

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
                        checklist_judgement=json.dumps(gpt_checklist_judgement, indent=4, ensure_ascii=False),
                    )
            # print(eval_prompt)
            conv.append({"role": "user", "content": eval_prompt})
            
            response = get_answer(
                    endpoint_info["model_name"],
                    conv,
                    configs["temperature"],
                    configs["max_tokens"],
                    args["endpoint_dict"],
                    require_json=True,
                )

            if response == '$ERROR$':
                print("API failed, output is ERROR!")
            elif response == '$REPETITIVE PATTERNS$':
                print("detect repetitive patterns")
                final_judge = {'API fialed': "$REPETITIVE PATTERNS$"}
                score = 0
                
            else:
                gpt_judge = check_gpt_judge(json.loads(response), configs['prompt_language'])
                
                if heuristic_checklist:
                    heuristic_judge = heuristic_judgement(cur_question, heuristic_checklist)
                else:
                    # print("There is no heuristic checklist")
                    heuristic_judge = {}
                    
                final_judge = {**gpt_judge, **heuristic_judge}
                score = get_score(final_judge, configs['prompt_language'])
                assert 0-1e-6<=score<=1+1e-6, f"score should be between 0 and 1, but got {score}, corrsponding to final_judge {final_judge}, corresponding to user_query {cur_question['user_query'][:10]}"
            
            output = {
                **cur_question,
                "judge_model": model,
                # "infer_model_in_judgepy": cur_question['infer_model'],
                # "second_resp_in_judgepy": cur_question['second_response'],
                # "eval_prompt": eval_prompt,
                "judgement": final_judge,
                "score": score 
                }
            
            with open(output_file, "a") as f:
                f.write(json.dumps(output, ensure_ascii=False) + "\n")
                # f.flush()
            
            
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
    answer_dir = os.path.join("data", configs["bench_name"], "model_answer")

    questions = load_data(question_file)
    model_answers = load_cache_data(answer_dir)
    # Organize the data in "model_answers" into a dictionary using "user_query" as the key.
    model_answers_dict = {model: {data_dict['user_query']: data_dict for data_dict in data} for model, data in model_answers.items()}
    
    # if user choose a set of models, only judge those models
    models = [model for model in configs["model_list"]]
    
    output_files = {}
    output_dir = f"data/{configs['bench_name']}/model_judgment/{configs['judge_model']}"
    for model in models:
        output_files[model] = os.path.join(
            output_dir,
            f"{model}.jsonl",
        )

    for output_file_path in output_files.values():
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    existing_judgments = load_cache_data(output_dir)

    endpoint_info = endpoint_list[configs["judge_model"]]


    with concurrent.futures.ThreadPoolExecutor(max_workers=endpoint_info["parallel"]) as executor:
        futures = []
        for model in models:
            count = 0
            for question in questions:
                
                kwargs = {}
                kwargs["question"] = copy.deepcopy(question)
                

                if model in model_answers and model_answers[model] and not question['user_query'] in set(data_dict['user_query'] for data_dict in model_answers[model]):
                    print(f"Warning: {model} answer to 【{question['user_query']}】 cannot be found.")
                    continue

                if model in existing_judgments and existing_judgments[model] and question['user_query'] in set(data_dict['user_query'] for data_dict in existing_judgments[model]):
                    count += 1
                    continue
                
                # Based on the model's name "h" and "user_query", find the corresponding "second_response" and "infer_model".
                if model in model_answers_dict and question['user_query'] in model_answers_dict[model]:
                    kwargs['question']['second_response'] = model_answers_dict[model][question['user_query']]['second_response']
                    kwargs['question']['infer_model'] = model_answers_dict[model][question['user_query']]['infer_model']
                    assert model == kwargs['question']['infer_model'], f"model name in model_answers_dict should be the same as model name in configs, but got {model} and {kwargs['question']['infer_model']}"
                else:
                    print(f"Warning: {model} answer to 【{question['user_query']}】 cannot be found.")
                    continue


                kwargs["configs"] = configs
                kwargs["endpoint_dict"] = endpoint_info
                kwargs["output_file"] = output_files[model]
                future = executor.submit(judgment, **kwargs)
                futures.append(future)

            if count > 0:
                print(f"{model} {count} number of existing judgments")

        for future in tqdm(
            concurrent.futures.as_completed(futures), total=len(futures)
        ):
            future.result()

    # Reorder the judgement results of the model.
    for model in models:
        output_file = output_files[model]
        reorg_file(output_file, sort_key="user_query")
        