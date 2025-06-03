"""Generate answers using api endpoints.

Usage:
python gen_api_answer --parallel 32
"""
import argparse
import json
import os
import time
import concurrent.futures

import tiktoken
# import shortuuid
import tqdm

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


def get_answer(
    question: dict, model: str, endpoint_info: dict, max_tokens: int, temperature: float, answer_file: str, api_dict: dict
):
    """Here, "question" is a dictionary that contains all the information about the question, including the user's query, the user's feedback, and so on."""
    
    if question["task_type"] in temperature_config:
        temperature = temperature_config[question["task_type"]]

    api_type = endpoint_info["api_type"]

    conv = []
    
    # load system prompt
    if "system_prompt" in endpoint_info.keys():
        conv.append({"role": "system", "content": endpoint_info["system_prompt"]})
    elif model in ['qwen-max-0919', 'qwen-plus-0919', 'qwen-max-0428', 'deepseek-chat', 'DeepSeek-V2.5', 'qwen-max', 'gpt-4o-2024-05-13', 'gpt-4o-2024-08-06', 'gpt-4o-mini-2024-07-18']:
        conv.append({"role": "system", "content": "You are a helpful assistant."})

    # load prefixed context
    conv.append({"role": "user", "content": question['user_query']})
    conv.append({"role": "assistant", "content": question['origin_first_response']})
    conv.append({"role": "user", "content": question['feedback']})
    
    # generate seconod response
    if api_type == "anthropic":
        output = chat_completion_anthropic(model=endpoint_info["model_name"],
                                            messages=conv,
                                            temperature=temperature,
                                            max_tokens=max_tokens)
    elif api_type == "mistral":
        output = chat_completion_mistral(model=endpoint_info["model_name"],
                                            messages=conv,
                                            temperature=temperature,
                                            max_tokens=max_tokens)
    elif api_type == "gemini":
        output = http_completion_gemini(model=endpoint_info["model_name"],
                                        message=question["turns"][j]["content"],
                                        temperature=temperature,
                                        max_tokens=max_tokens)
    elif api_type == "azure":
        output = chat_completion_openai_azure(model=endpoint_info["model_name"],
                                                messages=conv,
                                                temperature=temperature,
                                                max_tokens=max_tokens,
                                                api_dict=api_dict)
    elif api_type == "cohere":
        output = chat_completion_cohere(model=endpoint_info["model_name"],
                                        messages=conv,
                                        temperature=temperature,
                                        max_tokens=max_tokens)
    elif api_type == "ernie":
        output = chat_completion_ernie(model=endpoint_info["model_name"], 
                                        messages=conv, 
                                        temperature=temperature, 
                                        max_tokens=max_tokens, 
                                        api_dict=api_dict)
    else:
        output = chat_completion_openai(model=endpoint_info["model_name"], 
                                        messages=conv, 
                                        temperature=temperature, 
                                        max_tokens=max_tokens, 
                                        api_dict=api_dict)
    if output == '$ERROR$':
        print("API failed, output is ERROR!")
        # return None

    # save data
    ans = {
        **question,
        "second_response": output,
        "infer_model": model,
        "tsamp": time.time(),
    }

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
    args = parser.parse_args()

    settings = make_config(args.setting_file)
    endpoint_list = make_config(args.endpoint_file)

    existing_answer = load_cache_data(os.path.join("data", settings["bench_name"], "model_answer"))
    
    print(settings)

    for model in settings["model_list"]:
        assert model in endpoint_list
        endpoint_info = endpoint_list[model]

        question_file = os.path.join("data", settings["bench_name"], settings["test_file_name"])
        questions = load_data(question_file)

        answer_file = os.path.join("data", settings["bench_name"], "model_answer", f"{model}.jsonl")
        print(f"Output to {answer_file}")

        if "parallel" in endpoint_info:
            parallel = endpoint_info["parallel"]
        else:
            parallel = 1

        # We want to maximizes the number of tokens generate per answer: max_tokens = specified token # - input tokens #
        if "tokenizer" in endpoint_info:
            question_list = [' '.join([question['user_query'], question['origin_first_response'], question['feedback']]) for question in questions]
            from transformers import AutoTokenizer
            
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            tokenizer = AutoTokenizer.from_pretrained(endpoint_info["tokenizer"], trust_remote_code=True)
            tokens = tokenizer(question_list)
            # max_tokens = [(settings["max_tokens"] - len(prompt) - 300) for prompt in tokens["input_ids"]]
            max_tokens = [min(settings["max_tokens"], endpoint_info["max_model_len"]-len(prompt)-100) for prompt in tokens['input_ids']]
            # max_tokens = [token if token>=1 else settings["max_tokens"] for token in max_tokens]
        else:
            max_tokens = [settings["max_tokens"]] * len(questions)
        if model=='qwen-max':
            max_tokens = [2000] * len(questions)
        # print(f"minimum max_tokens is {min(max_tokens)}, maximum max_tokens is{max(max_tokens)}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as executor:
            futures = []
            count = 0
            # 用user_query来作为id，判断是否已经生成过答案
            for index, question in enumerate(questions):
                if model in existing_answer and existing_answer[model] and question["user_query"] in set(data_dict['user_query'] for data_dict in existing_answer[model]):
                    count += 1
                    continue
                future = executor.submit(
                    get_answer,
                    question,
                    model,
                    endpoint_info,
                    max_tokens[index],
                    settings["temperature"],
                    answer_file,
                    get_endpoint(endpoint_info["endpoints"]),
                )
                futures.append(future)
            if count > 0:
                print(f"{count} number of existing answers")
            for future in tqdm.tqdm(
                concurrent.futures.as_completed(futures), total=len(futures)
            ):
                future.result()

        reorg_file(answer_file, sort_key="user_query")
