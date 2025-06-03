"""Generate answers using api endpoints.

Usage:
python gen_api_answer --parallel 32
"""
import argparse
import json
import os
import time
import concurrent.futures
# from run_local_model import call_model_service_infer_msg
# import tiktoken
# import shortuuid
import tqdm
from openai import OpenAI
from transformers import AutoTokenizer
from copy import deepcopy

from utils import (
    load_data,
    load_cache_data,
    make_config,
    get_endpoint,
    chat_completion_openai,
    chat_completion_anthropic,
    chat_completion_openai_azure,
    chat_completion_mistral,
    http_completion_gemini,
    chat_completion_cohere,
    chat_completion_ernie,
    reorg_file,
    temperature_config,
)

global client
global client_cot

PROMPT_DEEPCLAUDE="""Here's my original input request:\n```\n{original_content}\n```\n\nHere's my another model's reasoning process:\n{reasoning}\n\nBased on this reasoning, provide your response directly to me:"""


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



def get_answer(
    question: dict, model: str, answer_file: str, cot_model: str):
    """Here, "question" is a dictionary that contains all the information about the question, including the user's query, the user's feedback, and so on."""
    
    # if question["task_type"] in temperature_config:
    #     temperature = temperature_config[question["task_type"]]
    global client
    global client_cot
    conv = []
    
    # load system prompt
    conv.append({"role": "system", "content": "You are a helpful assistant."})
    # load prefixed context
    conv.append({"role": "user", "content": question['user_query']})
    conv.append({"role": "assistant", "content": question['origin_first_response']})
    conv.append({"role": "user", "content": question['feedback']})
    
    # output = call_model_service_infer_msg(conv, model)
    if client_cot:
        messages_str = ""
        for message in conv:
            messages_str += message["content"]+"\n"
        max_tokens = max(1, min(int((16384 - count_token_max(messages_str, [tokenizer, tokenizer_llama, tokenizer_mistral]))// 1.3), 10000))
        response_cot = client_cot.with_options(timeout=7 * 1000).chat.completions.create(
                                model=args.cot_model_id,  
                                messages=conv,
                                max_tokens=max_tokens,
                                temperature=0).to_dict()
        assert "choices" in response_cot
        result_cot = response_cot["choices"][0]["message"]["content"]
        reasoning_cot_model_content = response_cot["choices"][0]["message"]["reasoning_content"]
        conv[-1]["content"] = PROMPT_DEEPCLAUDE.format(original_content=conv[-1]["content"], reasoning=reasoning_cot_model_content)
    else:
        reasoning_cot_model_content = None
        result_cot = None


    messages_str = ""
    for message in conv:
        messages_str += message["content"]+"\n"
    max_tokens = max(1, int((16384 - count_token_max(messages_str, [tokenizer, tokenizer_llama, tokenizer_mistral]))// 1.1 - 100))

    response = client.chat.completions.create(
                                model=model,  
                                messages=conv,
                                max_tokens=max_tokens,
                                temperature=0).to_dict()
    assert "choices" in response
    output = response["choices"][0]["message"]["content"]
    if output.startswith("<answer>"):
        output = (output[len("<answer>"):]).lstrip()
    if "reasoning_content" in response["choices"][0]["message"]:
        reasoning = response["choices"][0]["message"]["reasoning_content"]
    else:
        reasoning = None

    if output == '$ERROR$':
        print("API failed, output is ERROR!")
        # return None

    # save data
    ans = {
        **question,
        "second_response": output,
        "infer_model": model,
        "tsamp": time.time(),
        "reasoning": reasoning,
    }

    if reasoning_cot_model_content:
        ans[cot_model + "_reasoning_cot_model"] = reasoning_cot_model_content
    if result_cot:
        ans[cot_model + "_result_cot_model"] = result_cot

    # import pdb;pdb.set_trace();
    os.makedirs(os.path.dirname(answer_file), exist_ok=True)
    with open(answer_file, "a", encoding='utf-8') as fout:
        fout.write(json.dumps(ans, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--setting-file", type=str, default="config/gen_answer_config.yaml"
    )
    parser.add_argument(
        "--endpoint-file", type=str, default="config/api_config.yaml"
    )
    parser.add_argument(
        "--model_id", type=str, default=""
    )
    parser.add_argument(
        "--model_url", type=str, default=""
    )
    parser.add_argument(
        "--batch_size", type=int, default=16
    )
    parser.add_argument('--cot_model_id', type=str, default="")
    parser.add_argument('--cot_model_url', type=str, default="")
    parser.add_argument("--out_dir", type=str, default="evaluation")

    args = parser.parse_args()
    global client
    global client_cot

    model_url = args.model_url
    print("加载模型服务地址：", model_url)
    client = OpenAI(base_url=model_url, api_key = "xxx", timeout=600)
    print("加载模型名称：", args.model_id)

    cot_model_url = args.cot_model_url
    if cot_model_url != "" and cot_model_url != "N/A":
        print("加载CoT模型服务地址：", cot_model_url)
        client_cot = OpenAI(base_url=cot_model_url, api_key = "xxx", timeout=7200)
        print("加载CoT模型名称：", args.cot_model_id)
    else:
        print("CoT模型服务地址为空，不使用CoT模型")
        client_cot = None

    settings = make_config(args.setting_file)
    endpoint_list = make_config(args.endpoint_file)
    out_dir = args.out_dir
    model = args.model_id
    cot_model = args.cot_model_id
    os.makedirs(out_dir, exist_ok=True)
    # existing_answer = load_cache_data(os.path.join("data", settings["bench_name"], "model_answer"))
    existing_answer = load_cache_data(out_dir)
    
    print(settings)

    # endpoint_info = endpoint_list[model]

    question_file = os.path.join("data", settings["bench_name"], settings["test_file_name"])
    questions = load_data(question_file)

    # answer_file = os.path.join("data", settings["bench_name"], "model_answer", f"{model}.jsonl")
    answer_file = os.path.join(out_dir, f"{model}.jsonl")
    print(f"Output to {answer_file}")
    parallel = int(args.batch_size)
    print("Parallel with {} threads".format(parallel))

    # We want to maximizes the number of tokens generate per answer: max_tokens = specified token # - input tokens #
    # if "tokenizer" in endpoint_info:
    #     question_list = [' '.join([question['user_query'], question['origin_first_response'], question['feedback']]) for question in questions]
    #     from transformers import AutoTokenizer
        
    #     os.environ["TOKENIZERS_PARALLELISM"] = "false"
    #     tokenizer = AutoTokenizer.from_pretrained(endpoint_info["tokenizer"], trust_remote_code=True)
    #     tokens = tokenizer(question_list)
    #     # max_tokens = [(settings["max_tokens"] - len(prompt) - 300) for prompt in tokens["input_ids"]]
    #     max_tokens = [min(settings["max_tokens"], endpoint_info["max_model_len"]-len(prompt)-100) for prompt in tokens['input_ids']]
    #     # max_tokens = [token if token>=1 else settings["max_tokens"] for token in max_tokens]
    # else:
    #     max_tokens = [settings["max_tokens"]] * len(questions)
    # if model=='qwen-max':
    #     max_tokens = [2000] * len(questions)
    # print(f"minimum max_tokens is {min(max_tokens)}, maximum max_tokens is{max(max_tokens)}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as executor:
        futures = []
        count = 0
        # 用user_query来作为id，判断是否已经生成过答案
        for index, question in enumerate(questions):
            if model in existing_answer and existing_answer[model] and question["user_query"] in set(data_dict['user_query'] for data_dict in existing_answer[model]):
                count += 1
                continue
            # question: dict, model: str, endpoint_info: dict, max_tokens: int, temperature: float, answer_file: str, api_dict: dict
            future = executor.submit(
                get_answer,
                question,
                model,
                answer_file,
                cot_model,
            )
            futures.append(future)
        if count > 0:
            print(f"{count} number of existing answers")
        for future in tqdm.tqdm(
            concurrent.futures.as_completed(futures), total=len(futures)
        ):
            future.result()

    reorg_file(answer_file, sort_key="user_query")
