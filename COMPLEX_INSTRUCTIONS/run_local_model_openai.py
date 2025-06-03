import os, json, requests, re, traceback
from flask import Flask, request, jsonify, stream_with_context, Response
from queue import Queue, Empty
import argparse
import threading
import asyncio
import aiohttp
import time
import pandas as pd
import xlsxwriter
from tqdm import tqdm
from datetime import datetime
import _run_multi_urls
from importlib import reload
import logging
import concurrent.futures
from openai import OpenAI
from transformers import AutoTokenizer


parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type=str, default="")
parser.add_argument('--input_file_type', type=str, default="jsonl")
parser.add_argument('--save_path', type=str, default="")
parser.add_argument('--model_id', type=str, default="")
parser.add_argument('--model_url', type=str, default="")
parser.add_argument('--cot_model_id', type=str, default="")
parser.add_argument('--cot_model_url', type=str, default="")

parser.add_argument('--question_type', type=str, default="conversations") #! TODO
parser.add_argument('--save_freq', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=8)
# parser.add_argument('--custom_prompt', action="store_true", default=False)
parser.add_argument('--qkey', type=str, default='q') # 输入文件种query对应的key
parser.add_argument('--akey', type=str, default='default') # 输出文件中answer对应的key
parser.add_argument('--system_prompt', type=str, default='You are a helpful assistant.')
# parser.add_argument('--answer_starts_with', type=str, default='')
parser.add_argument('--temperature', type=float, default=0.)
parser.add_argument('--max_tokens', type=int, default=512)
parser.add_argument('--top_k', type=int, default=1)
parser.add_argument('--top_p', type=float, default=1.0)
parser.add_argument('--enable_token_count', type=str, default='false') #! TODO
parser.add_argument('--resume', action="store_true", default=True)
parser.add_argument('--api_key', type=str, default="xxx")

args = parser.parse_args()
global client
global client_cot


PROMPT_DEEPCLAUDE="""Here's my original input request:\n```\n{original_content}\n```\n\nHere's my another model's reasoning process:\n{reasoning}\n\nBased on this reasoning, provide your response directly to me:"""

# PROMPT_DEEPCLAUDE="""\n\n<think>\n{reasoning}\n</think>"""


def process_streaming(response):
    start_time = time.time()
    collected_chunks = []
    collected_messages = []
    # iterate through the stream of events
    for chunk in response:
        chunk_time = time.time() - start_time  # calculate the time delay of the chunk
        collected_chunks.append(chunk)  # save the event response
        chunk_message = chunk.choices[0].delta.content  # extract the message
        collected_messages.append(chunk_message)  # save the message
        # print(f"Message received {chunk_time:.2f} seconds after request: {chunk_message}")  # print the delay and text

    # print the time delay and text received
    print(f"Full response received {chunk_time:.2f} seconds after request")
    # clean None in collected_messages
    collected_messages = [m for m in collected_messages if m is not None]
    full_reply_content = ''.join(collected_messages)
    # print(f"Full conversation received: {full_reply_content}")
    return full_reply_content




def build_model():
    model_name_or_path = "{YOUR_PATH_TO_PRETRAINED_MODELS}/pretrained_models/Qwen2.5-7B-Instruct_Qwen"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
    print("loaded tokenizer qwen")
    return tokenizer


def build_model_mistral():
    model_name_or_path = "{YOUR_PATH_TO_PRETRAINED_MODELS}/pretrained_models/Mistral-7B-Instruct-v0.3"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
    print("loaded tokenizer mistral")
    return tokenizer


def build_model_llama():
    model_name_or_path = "{YOUR_PATH_TO_PRETRAINED_MODELS}/pretrained_models/Meta-Llama-3.1-8B-Instruct_meta-llama"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
    print("loaded tokenizer llama")
    return tokenizer


def count_token(response, tokenizer):
    # try:
    #     inputs = tokenizer.apply_chat_template(response, tokenize=True)
    # except:
    #     inputs = tokenizer.encode(response)
    inputs = tokenizer.encode(response)
    
    return len(inputs)


def count_token_max(response, tokenizer_list):
    response_token_len_list = [count_token(response, tokenizer_item) for tokenizer_item in tokenizer_list]
    return max(response_token_len_list)


global tokenizer
if "mistral" in (args.model_id).lower() or "mistral" in (args.cot_model_id).lower() or "ministral" in (args.model_id).lower():
    tokenizer = build_model_mistral()
elif "llama" in (args.model_id).lower() or "llama" in (args.cot_model_id).lower():
    tokenizer = build_model_llama()
else:
    tokenizer = build_model()


if args.question_type == 'conversations':
    args.qkey = "conversations"


def call_model_service(index, data):
    global client
    global tokenizer
    global client_cot
    if "tools" in data:
        tools = json.loads(data["tools"])
    else:
        tools = None

    if args.question_type == 'conversations':
        messages = data[args.qkey]
    else:
        messages = [
            {
                "role":"system",
                "content":args.system_prompt,
            },
            {
                "role":"user",
                "content":data[args.qkey],
            }
        ]

    messages_new = []
    is_mistral = ("mistral" in str(args.model_id).lower())

    for message in messages:
        role = message["role"]
        content = message["content"]

        if is_mistral and not (role in ["user", "assistant", "system"]):
            if messages_new[-1]["role"] == "user":
                messages_new[-1]["content"] += f"\n[{role}]\n{content}\n[/{role}]"
                continue

            else:
                content = f"\n[{role}]\n{content}\n[/{role}]"
                role = "user"

        messages_new.append(
            {
                "role":role,
                "content":content,
            }
        )
    while messages_new[-1]["role"] == "assistant":
        messages_new.pop(-1)
        
    messages = messages_new
    messages_str = ""
    for message in messages:
        if message["content"] is not None:
            messages_str += message["content"]+"\n"

        if "tool_calls" in message and message["tool_calls"] is not None:
            messages_str += str(message["tools"])+"\n"

    # if count_token_max(messages_str, [tokenizer, tokenizer_llama, tokenizer_mistral]) >= (16384-args.max_tokens)/1.3:
    #     print("Overlength inputs")
    #     if args.akey == 'default':
    #         data[args.model_id] = "ERROR:OVERLENGTH"
    #     else:
    #         data[args.akey] = "ERROR:OVERLENGTH"
    #     return index, data

    num_retry = 0
    use_stream = False  
    # use_stream = True

    while num_retry < 10:
        # if True:
        try:
            if client_cot:
                messages_str = ""
                for message in messages:
                    messages_str += message["content"]+"\n"
                # max_tokens = max(1, min(int((16384 - count_token_max(messages_str, [tokenizer, tokenizer_mistral, tokenizer_llama]))// 1.4), 10000))
                max_tokens = max(1, min(int((16384 - count_token(messages_str, tokenizer))// 1.4), 10000))

                response_cot = client_cot.with_options(timeout=7 * 1000).chat.completions.create(
                                        model=args.cot_model_id,  
                                        messages=messages,
                                        max_tokens=max_tokens,
                                        temperature=args.temperature,
                                        stream=use_stream,
                                        # extra_body={"priority": 0}
                                        )
                
                if use_stream:
                    result_cot = process_streaming(response_cot)

                else:
                    response_cot = response_cot.to_dict()
                    assert "choices" in response_cot
                    result_cot = response_cot["choices"][0]["message"]["content"]
                    reasoning_cot_model_content = response_cot["choices"][0]["message"]["reasoning_content"]
                
                messages[-1]["content"] = PROMPT_DEEPCLAUDE.format(original_content=messages[-1]["content"], reasoning=reasoning_cot_model_content)

            else:
                reasoning_cot_model_content = None
                result_cot = None

            messages_str = ""
            for message in messages:
                if message["content"] is not None:
                    messages_str += message["content"]+"\n"
                    
                if "tool_calls" in message and message["tool_calls"] is not None:
                    messages_str += str(message["tools"])+"\n"
            # max_tokens = max(int((16384 - count_token_max(messages_str, [tokenizer, tokenizer_llama, tokenizer_mistral]))// 1.3 - 100), 1)
            max_tokens = max(int((16384 - count_token(messages_str, tokenizer))// 1.3 - 100), 1)
            # max_tokens = 5
            temperature = args.temperature
            if num_retry > 5:
                temperature = 1
            
            generation_kwargs = {
                "max_tokens":max_tokens,
                "temperature":temperature,
                "stream":use_stream,
            }
            # if tools:
            #     generation_kwargs["tools"] = tools
            # if args.top_p != 1:
            #     generation_kwargs["top_p"] = args.top_p

            # print("generation_kwargs", generation_kwargs)
            response = client.with_options(timeout=7 * 100000).chat.completions.create(
                                        model=args.model_id,  
                                        messages=messages,
                                        **generation_kwargs
                                        )
            
            if use_stream:
                result = process_streaming(response)
            else:
                response = response.to_dict()
                assert "choices" in response
                result = response["choices"][0]["message"]["content"]
                # import pdb;pdb.set_trace();
                assert (result is not None) or ("tool_calls" in response["choices"][0]["message"] and response["choices"][0]["message"]["tool_calls"] is not None)

            if result and result.startswith("<answer>"):
                result = (result[len("<answer>"):]).lstrip()
            if "reasoning_content" in response["choices"][0]["message"]:
                reasoning = response["choices"][0]["message"]["reasoning_content"]
            else:
                reasoning = None
            
            if "tool_calls" in response["choices"][0]["message"]:
                tool_calls = response["choices"][0]["message"]["tool_calls"]
            else:
                tool_calls = None

            if args.akey == 'default':
                data[args.model_id] = result
                if reasoning:
                    data[args.model_id + "_reasoning"] = reasoning
                if tool_calls:
                    data[args.model_id + "_tool_calls"] = tool_calls

            else:
                data[args.akey] = result
                if reasoning:
                    data[args.akey + "_reasoning"] = reasoning
                if tool_calls:
                    data[args.model_id + "_tool_calls"] = tool_calls

            if reasoning_cot_model_content:
                data[args.cot_model_id + "_reasoning_cot_model"] = reasoning_cot_model_content
            if result_cot:
                data[args.cot_model_id + "_result_cot_model"] = result_cot

            return index, data

        except Exception as e:
            print("error", e)
            print('!'*50)
            num_retry += 1
            time.sleep(5)
            continue

    if args.model_id == "HY70B_32wT_250104":
        result = "INVALID RESPONSE"
        if args.akey == 'default':
            data[args.model_id] = result
        else:
            data[args.akey] = result
        return index, data   
         
    assert (0 == 1)
    return



def save_results(results, save_path):
    """
    分批次保存输出结果, a+ mode
    results: [index, result]
    """
    with open(save_path, 'a+') as f:
        for results_i in results:
            f.writelines(json.dumps(results_i[1], ensure_ascii=False)+'\n')


def load_input_file(input_path, file_type='jsonl'):
    if file_type == 'jsonl':
        with open(args.input_path) as f:
            Data = f.readlines()
        Data = [json.loads(i) for i in Data]
    elif file_type == 'json':
        with open(args.input_path) as f:
            Data = json.load(f)
    else:
        raise ValueError('file_type must be jsonl or json')
    if args.question_type == 'conversations':
        for i in range(len(Data)):
            #! 删掉最后的assistant
            if Data[i][args.qkey][-1]['role'] == 'assistant':
                Data[i][args.qkey].pop(-1)

    return Data


def get_resume_state(save_path, file_type='jsonl'):
    count = 0
    # 如果文件不存在
    if not os.path.exists(save_path):
        return count
    # 如果文件存在，加载处理进度
    if file_type == 'jsonl':
        with open(save_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line:
                    count += 1
    elif file_type == 'json':
        with open(save_path, 'r', encoding='utf-8') as f:
            count = len(json.load(f))
    else:
        raise ValueError('file_type must be jsonl or json')
    return count

    

def main():
    global client
    global client_cot

    if args.save_path.lower() == "default":
        default_output_dir = "/apdcephfs_cq8/share_2992827/shennong_5/_TEST/test_data_output"
        time_stamp = datetime.now().strftime("%Y_%m%d_%H%M%S")
        date_stamp = datetime.now().strftime("%Y-%m-%d")
        args.save_path = os.path.join(default_output_dir, date_stamp, os.path.basename(args.input_path).split('.')[0] + f'_{args.model_id}_ans.jsonl')

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    if args.resume:
        resume_state = get_resume_state(args.save_path, args.input_file_type)
    else:
        resume_state = 0

    # 加载输入文件
    Data = load_input_file(args.input_path, args.input_file_type)
    print(f"Data loaded. Total {len(Data)} records. {len(Data)-resume_state} records to run.")
    # cut Data
    Data = Data[resume_state:]

    # 加载模型服务地址
    model_url = args.model_url
    print("加载模型服务地址：", model_url)
    client = OpenAI(base_url=model_url, api_key=args.api_key, timeout=7200)
    print("加载模型名称：", args.model_id)

    cot_model_url = args.cot_model_url
    if cot_model_url != "" and cot_model_url != "N/A":
        print("加载CoT模型服务地址：", cot_model_url)
        client_cot = OpenAI(base_url=cot_model_url, api_key=args.api_key, timeout=7200)
        print("加载CoT模型名称：", args.cot_model_id)
    else:
        print("CoT模型服务地址为空，不使用CoT模型")
        client_cot = None

    #! TODO
    # 初始化缓存队列
    data_queue = Queue()
    for i, data in enumerate(Data):
        data_queue.put((i, data))

    # 初始化杂项
    results = {} # 输出缓存
    saved_count = 0  # 记录当前进度
    continuous_results = [] # 待保存的连续输出结果序列
    tbar = tqdm(total=len(Data)+resume_state, initial=resume_state, desc="Processing")

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.batch_size) as executor:
        futures = {executor.submit(call_model_service, *data_queue.get()): _ for _ in range(min(args.batch_size, data_queue.qsize()))}

        while futures:
            # 等待任何一个任务完成
            done, futures = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)

            # 同步已完成的任务
            for future in done:
                index, result = future.result()  # 获取index和结果
                results[index] = result # 将结果添加到输出缓存
                
                # 从缓存中取出连续的结果
                for i in range(saved_count, len(Data)):
                    if i in results:
                        continuous_results.append([i, results[i]])
                        saved_count += 1
                        tbar.update(1)
                        del results[i]
                    else:
                        break

                # 保存连续的结果
                if len(continuous_results) >= args.save_freq:
                    save_results(continuous_results, args.save_path)
                    # 重置连续结果序列
                    continuous_results = []
                
                # 向线程池中补充新任务
                try:
                    new_data = data_queue.get_nowait()
                    new_future = executor.submit(call_model_service, *new_data)
                    futures.add(new_future)
                # 直到数据全跑完
                except Empty:
                    pass

    # 保存剩余的结果
    # 清空连续结果序列
    if continuous_results:
        save_results(continuous_results, args.save_path)
    # 清空缓存
    if results:
        continuous_results = []
        for index in sorted(list(results.keys())):
            continuous_results.append([index, results[index]])
            del results[index]
            tbar.update(1)
        save_results(continuous_results, args.save_path)
    return


if __name__ == "__main__":
    main()
