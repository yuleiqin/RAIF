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

# parser = argparse.ArgumentParser()
# parser.add_argument('--input_path', type=str, default="")
# parser.add_argument('--input_file_type', type=str, default="jsonl")
# parser.add_argument('--save_path', type=str, default="")
# parser.add_argument('--model_id', type=str, default="")
# parser.add_argument('--question_type', type=str, default="conversations") #! TODO
# parser.add_argument('--save_freq', type=int, default=10)
# parser.add_argument('--batch_size', type=int, default=8)
# # parser.add_argument('--custom_prompt', action="store_true", default=False)
# parser.add_argument('--qkey', type=str, default='q')
# parser.add_argument('--system_prompt', type=str, default='')
# # parser.add_argument('--answer_starts_with', type=str, default='')
# parser.add_argument('--temperature', type=float, default=0.)
# parser.add_argument('--max_tokens', type=int, default=0)
# parser.add_argument('--top_k', type=int, default=0)
# parser.add_argument('--enable_token_count', type=str, default='false') #! TODO
# parser.add_argument('--resume', action="store_true", default=True)
# args = parser.parse_args()
# global model_url
# if args.question_type == 'conversations':
#     args.qkey = "conversations"


# save_path = 'output_data/results.json'

# def get_model_response(data):
#     global model_url
#     # url = "http://9.91.12.52:8001/forward"
#     # try:
#     response = requests.post(model_url, json=data)
#     # except:
#     # print('='*50)
#     # print(response.content)
#     # print(response.text)
#     #     return {"status": False, "response": '调用失败, IP:{}'.format(url)}
#     return {"status": True, "response": json.loads(response.text)['response']}

def get_model_response(data, model_url):
    response = requests.post(model_url, json=data)
    # import pdb;pdb.set_trace();
    return {"status": True, "response": json.loads(response.text)['response']}

# 获取模型返回，并更新到原dict中
def special_rules(conversation_i):
    # 删除神农prompt, 删除You are a helpful assistant
    if conversation_i[0]['role'] == 'system' and '神农' in conversation_i[0]['content']:
        conversation_i.pop(0)
    if conversation_i[0]['role'] == 'system' and conversation_i[0]['content'] == 'You are a helpful assistant':
        conversation_i.pop(0)
    return conversation_i

# def call_model_service(index, data):
#     if args.question_type == 'conversations':
#         # post_info = {"prompt": special_rules(data[args.qkey]), "messages_format": True, 'system': args.system_prompt}
#         post_info = {"prompt": data[args.qkey], "messages_format": True, 'system': args.system_prompt}
#     else:
#         post_info = {"prompt": data[args.qkey], 'system': args.system_prompt}

#     if args.temperature: post_info['temperature'] = args.temperature
#     if args.max_tokens: post_info['max_tokens'] = args.max_tokens
#     if args.top_k: post_info['top_k'] = args.top_k
#     result = get_model_response(post_info)
#     data[args.model_id] = result['response']
#     return index, data


def call_model_service_infer(messages, model_id, maxtry=10):
    model_url = load_model_url(model_id)
    assert(messages[0]["role"] == "system")
    assert(messages[1]["role"] == "user")
    post_info = {"prompt": messages[1]["content"], 'system': messages[0]["content"]}
    post_info['temperature'] = 0
    post_info['top_k'] = 1
    result = get_model_response(post_info, model_url)
    response = result['response']
    return response


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

def load_model_url(model_id):
    try:
        model_url = _run_multi_urls.local_urls[model_id]
    except Exception as e:
        raise ValueError(f'model_id {model_id} not found in local_urls')
    return model_url


def get_resume_state(save_path, file_type='jsonl'):
    count = 0
    # 如果文件不存在
    if not os.path.exists(save_path):
        return count
    # 如果文件存在，加载处理进度
    if file_type == 'jsonl':
        with open(save_path, 'r') as f:
            for line in f:
                if line:
                    count += 1
    elif file_type == 'json':
        with open(save_path, 'r') as f:
            count = len(json.load(f))
    else:
        raise ValueError('file_type must be jsonl or json')
    return count

    

def main(messages):
    global model_url

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
    model_url = load_model_url(args.model_id)

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

if __name__ == "__main__":
    main()
