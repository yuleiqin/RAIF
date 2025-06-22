import logging
import time
from abc import ABC
from copy import deepcopy
from dataclasses import dataclass
from typing import Generator, List, Optional, Tuple, Union
import numpy as np
import ray
import torch
import torch.nn as nn
from tqdm import tqdm
import torch
import torch.distributed as dist
import torch.nn.functional as F
import os
from openrlhf.models.actor import Actor
from openrlhf.models.utils import compute_approx_kl, compute_reward, masked_mean, unpacking_samples
from openrlhf.utils.logging_utils import init_logger
from openrlhf.utils.remote_rm_utils import remote_rm_fn, remote_rm_fn_ray
#from math_equal_file import math_equal
from openrlhf.trainer.ppo_utils.math_equal_file import math_equal
from ray.exceptions import GetTimeoutError
from openrlhf.trainer.ppo_utils.qwen_math_eval_toolkit.parser import extract_answer as qwen_extract_answer
logger = init_logger(__name__)
from openrlhf.trainer.ppo_utils.kk import compute_score as compute_kk_score
from openrlhf.trainer.ppo_utils.kk import validate_response_structure
from openrlhf.trainer.ppo_utils.ifeval.check_qa import compute_ifeval_score
from openrlhf.trainer.ppo_utils.ifeval.instructions_util import LANGUAGE_CODES
from openrlhf.trainer.ppo_utils.complex_scoring import compute_judgement_score_preprocess
from openrlhf.reward.repetition import get_repetition_penalty, get_repetition_penalty_tokens
from transformers import AutoTokenizer
from openrlhf.trainer.ppo_utils.complex_scoring import prepare_complex_judgement, compute_judgement_score_postprocess
# from openrlhf.trainer.ppo_utils.complex_scoring import prepare_complex_judgement_all_in_one as prepare_complex_judgement
# from openrlhf.trainer.ppo_utils.complex_scoring import compute_judgement_score_postprocess_all_in_one as compute_judgement_score_postprocess

from copy import deepcopy
import time
import re
import json
import langdetect


def preprocess_orm800k_response(sequence):
    temp_query = ""
    temp_response = ""

    temp_query = sequence.split("ASSISTANT:\n")[0]
    #temp_query = temp_query.split("# Question\n\n")[-1]

    temp_response = sequence.split("ASSISTANT:\n")[-1]
    
    #print("temp_response", temp_response)
    # 使用正则表达式匹配
    # pattern = r"The final answer is: \\\\boxed\{(.*?)\}"
    # # 正则表达式，处理换行与特殊字符
    # pattern = r"The final answer is: \\\\boxed\{(.*?)\}"
    #pattern = r"The final answer is: \\\\boxed\{(.*?)\}"
    pattern = r"The final answer is: \\boxed\{(.*?)\}"
    # 使用 re.DOTALL 确保能匹配跨行文本
    match = re.search(pattern, temp_response, re.DOTALL)
    #match = re.search(pattern, temp_response)
    if match:
        temp_answer = match.group(1)
    else:
        temp_answer = "none"

    #temp_answer = temp_response.split("\n\n# Answer\n\n")[-1]
    temp_response = sequence.split("<|reserved_special_token_0|>The final answer is:")[0]

    #response_list = temp_response.split("<|reserved_special_token_0|>")

    processed_solution = temp_response + "\n\n# Answer\n\n" + temp_answer + "<|reserved_special_token_0|>"

    # for i in range(len(response_list)-1):
    #     processed_solution = processed_solution + "Step " + str(i+1) + ": " + response_list[i] + " <|reserved_special_token_0|>\n"

    # processed_solution = processed_solution + response_list[-1] + " The answer is: " + temp_answer + " <|reserved_special_token_0|>\n"
    processed_solution = re.sub(r"<\|end_of_text\|>", "", processed_solution)
    return processed_solution


def compute_step_reward(
    r: Union[torch.Tensor, float],
    kl_coef: float,
    kl: Union[torch.Tensor, list[torch.Tensor]],
    reward_indexs: list,
    step_rewards: list,
    action_mask: Optional[torch.Tensor] = None,
    num_actions: Optional[Union[int, list[int]]] = None,
    reward_clip_range: Tuple[float, float] = None,
) -> Union[torch.Tensor, list[torch.Tensor]]:
    # print("step_rewards", step_rewards)
    # print("r", r)
    if kl_coef <= 0.0:
        kl_coef = 0.0

    if reward_clip_range:
        r = r.clamp(min=reward_clip_range[0], max=reward_clip_range[1])
        step_rewards = [tensor.clamp(min=reward_clip_range[0], max=reward_clip_range[1]) for tensor in step_rewards]

    if action_mask is not None:
        kl_reward = -kl_coef * kl
        
        # print("kl_reward.shape", kl_reward.shape)
        # #print("kl_reward", kl_reward)
        # print("action_mask.size", action_mask.size(1))
        # The following code is equivalent to:
        #
        # last_reward = torch.zeros_like(kl)
        # for i in range(last_reward.size(0)):
        #     for t in reversed(range(last_reward.size(1))):
        #         if action_mask[i][t] > 0.5:
        #             last_reward[i][t] = r[i]
        #             break
        #
        eos_indices = action_mask.size(1) - 1 - action_mask.long().fliplr().argmax(dim=1, keepdim=True)
        # print("step_rewards", step_rewards)
        # print("reward_indexs", reward_indexs)
        # print("r", r)
        # print("r.shape", r.shape)
        # #print("last_reward", last_reward)
        # #print("last_reward.shape", last_reward.shape)
        
        # print("kl_reward", kl_reward)
        
        # print("kl_reward.shape", kl_reward.shape)
        final_reward = kl_reward.clone()
        #final_reward = torch.tensor(f)
        last_reward = torch.zeros_like(kl).scatter_(dim=1, index=eos_indices, src=r.unsqueeze(1).to(kl.dtype))
        
        # 遍历 step_rewards 和 reward_indexs
        for i, (step_reward, reward_index) in enumerate(zip(step_rewards, reward_indexs)):
            #print("reward_index", reward_index)
            #print("final_reward.shape", final_reward.shape)
            
            reward_index = [reward_i + action_mask.size(1) for reward_i in reward_index]
            reward_index = torch.tensor(reward_index, device=kl_reward.device)
            #reward_index = reward_index + action_mask.size(1) 
            
            # print("pos_reward_index", reward_index)
            # print("reward_index.max().item()", reward_index.max().item())
            # print("eos_indices", eos_indices)
            # print("eos_indices", eos_indices[i])
            
            # # print("final_reward[i, reward_index].shape", final_reward[i, reward_index].shape)
            # # print("step_reward.shape", step_reward.shape)
            # # print("final_reward[i, :]", final_reward[i, :])
            # # print("last_reward[i, :]", last_reward[i, :])
            # print("final_reward[i, :]", final_reward[i, :])
            if reward_index.numel() > 0 and eos_indices.numel() > 0: 
                #print("reward_index", reward_index)
                #print("final_reward.shape", final_reward.shape)
                #print("action_mask.size(1)", action_mask.size(1))
                #print("final_reward", final_reward)
                
                if reward_index.max().item() < final_reward.shape[1]:
            
                    #if reward_index[-1] + 1 == eos_indices[-1]:
                    if final_reward[i, reward_index].shape == step_reward.shape:
                        final_reward[i, reward_index] += (step_reward) / step_reward.shape[-1]
                        #print("1")
                    else:
                        final_reward[i, :] = last_reward[i, :]
                        #print("2")
                    #else:
                    #    final_reward[i, :] = last_reward[i, :]
                        
                    
                # print("final_reward[i, reward_index].shape", final_reward[i, reward_index].shape)
                # print("step_reward.shape", step_reward.shape)
                # print("final_reward[i, reward_index]", final_reward[i, reward_index])
                # print("step_reward", step_reward)
                else:
                    final_reward[i, :] = last_reward[i, :]
                    #print("2")
                    
            else:
                final_reward[i, :] = last_reward[i, :]
                #print("2")
                
                
                                  
                
            # # 直接用负索引将 step_reward 加到 last_reward 对应位置
            # if final_reward[i, reward_index].shape == step_reward.shape:
            #     final_reward[i, reward_index] += (step_reward) / step_reward.shape[-1]
            #     print("final_reward[i, reward_index].shape", final_reward[i, reward_index].shape)
            #     print("step_reward.shape", step_reward.shape)
            #     print("final_reward[i, reward_index]", final_reward[i, reward_index])
            #     print("step_reward", step_reward)                 
            # else:
            #     final_reward[i, :] = last_reward[i, :]
            #     #print("final_reward[i, :]", final_reward[i, :])
            #     print("last_reward[i, :]", last_reward[i, :])
                
            
            # print("last_reward[i, reward_index]", step_reward[i, reward_index])
            # print("step_reward", step_reward)
            # step_reward[i, reward_index] += step_reward
            
        
        final_reward = torch.tensor(final_reward, device=kl_reward.device)
        #last_reward = torch.zeros_like(kl).scatter_(dim=1, index=eos_indices, src=r.unsqueeze(1).to(kl.dtype))
        # print("final_reward", final_reward)
        # print("final_reward.shape", final_reward.shape)
        
        # # 遍历 step_rewards 和 reward_indexs
        # for i, (step_reward, reward_index) in enumerate(zip(step_rewards, reward_indexs)):
        #     # 直接用负索引将 step_reward 加到 kl_reward 对应位置
        #     kl_reward[i, reward_index] += step_reward
        
        
        reward = final_reward
        #reward = last_reward + kl_reward
    else:
        # TODO: write a more efficient version
        reward = []
        for i, (kl_seg, action_len) in enumerate(zip(kl, num_actions)):
            kl_reward = -kl_coef * kl_seg
            kl_reward[action_len - 1] += r[i]
            reward.append(kl_reward)

    return reward


def preprocess_orm_response(sequence):
    temp_query = ""
    temp_response = ""

    temp_query = sequence.split("\n\n# Solution\n\n")[0]
    temp_query = temp_query.split("# Question\n\n")[-1]

    temp_response = sequence.split("\n\n# Solution\n\n")[-1]

    temp_answer = temp_response.split("\n\n# Answer\n\n")[-1]
    temp_response = temp_response.split("\n\n# Answer\n\n")[0]

    response_list = temp_response.split("\n\n")

    processed_solution = ""

    for i in range(len(response_list)-1):
        processed_solution = processed_solution + "Step " + str(i+1) + ": " + response_list[i] + " <|reserved_special_token_0|>\n"

    processed_solution = processed_solution + response_list[-1] + " The answer is: " + temp_answer + " <|reserved_special_token_0|>\n"

    return temp_query + " " + processed_solution


def preprocess_orm800k_response(sequence):
    temp_query = ""
    temp_response = ""

    temp_query = sequence.split("ASSISTANT:\n")[0]
    #temp_query = temp_query.split("# Question\n\n")[-1]

    temp_response = sequence.split("ASSISTANT:\n")[-1]
    
    #print("temp_response", temp_response)
    # 使用正则表达式匹配
    # pattern = r"The final answer is: \\\\boxed\{(.*?)\}"
    # # 正则表达式，处理换行与特殊字符
    # pattern = r"The final answer is: \\\\boxed\{(.*?)\}"
    #pattern = r"The final answer is: \\\\boxed\{(.*?)\}"
    pattern = r"The final answer is: \\boxed\{(.*?)\}"
    # 使用 re.DOTALL 确保能匹配跨行文本
    match = re.search(pattern, temp_response, re.DOTALL)
    #match = re.search(pattern, temp_response)
    if match:
        temp_answer = match.group(1)
    else:
        temp_answer = "none"

    #temp_answer = temp_response.split("\n\n# Answer\n\n")[-1]
    temp_response = sequence.split("<|reserved_special_token_0|>The final answer is:")[0]

    #response_list = temp_response.split("<|reserved_special_token_0|>")

    processed_solution = temp_response + "\n\n# Answer\n\n" + temp_answer + "<|reserved_special_token_0|>"

    # for i in range(len(response_list)-1):
    #     processed_solution = processed_solution + "Step " + str(i+1) + ": " + response_list[i] + " <|reserved_special_token_0|>\n"

    # processed_solution = processed_solution + response_list[-1] + " The answer is: " + temp_answer + " <|reserved_special_token_0|>\n"
    processed_solution = re.sub(r"<\|end_of_text\|>", "", processed_solution)
    return processed_solution


def preprocess_orm800k_box_response(sequence, answer):
    temp_query = ""
    temp_response = ""

    temp_query = sequence.split("ASSISTANT:\n")[0]
    #temp_query = temp_query.split("# Question\n\n")[-1]

    temp_response = sequence.split("ASSISTANT:\n")[-1]
    
    #print("temp_response", temp_response)
    # 使用正则表达式匹配
    # pattern = r"The final answer is: \\\\boxed\{(.*?)\}"
    # # 正则表达式，处理换行与特殊字符
    # pattern = r"The final answer is: \\\\boxed\{(.*?)\}"
    #pattern = r"The final answer is: \\\\boxed\{(.*?)\}"
    pattern = r"The final answer is: \\boxed\{(.*?)\}"
    # 使用 re.DOTALL 确保能匹配跨行文本
    match = re.search(pattern, temp_response, re.DOTALL)
    #match = re.search(pattern, temp_response)
    if match:
        temp_answer = match.group(1)
    else:
        temp_answer = "none"

    #temp_answer = temp_response.split("\n\n# Answer\n\n")[-1]
    temp_response = sequence.split("<|reserved_special_token_0|>The final answer is:")[0]

    #response_list = temp_response.split("<|reserved_special_token_0|>")

    processed_solution = temp_response + "\n\n# Answer\n\n" + temp_answer + "<|reserved_special_token_0|>"

    # for i in range(len(response_list)-1):
    #     processed_solution = processed_solution + "Step " + str(i+1) + ": " + response_list[i] + " <|reserved_special_token_0|>\n"

    # processed_solution = processed_solution + response_list[-1] + " The answer is: " + temp_answer + " <|reserved_special_token_0|>\n"
    processed_solution = re.sub(r"<\|end_of_text\|>", "", processed_solution)
    
    if temp_answer == answer:
        box_match = 1.0
    else:
        box_match = 0.0
    return processed_solution, box_match



def preprocess_orm800k_box_responsev1(sequence, answer):
    temp_query = ""
    temp_response = ""

    temp_query = sequence.split("ASSISTANT:\n")[0]
    #temp_query = temp_query.split("# Question\n\n")[-1]

    temp_response = sequence.split("ASSISTANT:\n")[-1]
    
    #print("temp_response", temp_response)
    # 使用正则表达式匹配
    # pattern = r"The final answer is: \\\\boxed\{(.*?)\}"
    # # 正则表达式，处理换行与特殊字符
    # pattern = r"The final answer is: \\\\boxed\{(.*?)\}"
    #pattern = r"The final answer is: \\\\boxed\{(.*?)\}"
    pattern = r"The final answer is: \\boxed\{(.*?)\}"
    # 使用 re.DOTALL 确保能匹配跨行文本
    match = re.search(pattern, temp_response, re.DOTALL)
    #match = re.search(pattern, temp_response)
    if match:
        temp_answer = match.group(1)
    else:
        temp_answer = "none"

    #temp_answer = temp_response.split("\n\n# Answer\n\n")[-1]
    temp_response = sequence.split("<|reserved_special_token_0|>The final answer is:")[0]

    #response_list = temp_response.split("<|reserved_special_token_0|>")

    processed_solution = temp_response + "\n\n# Answer\n\n" + temp_answer + "<|reserved_special_token_0|>"

    # for i in range(len(response_list)-1):
    #     processed_solution = processed_solution + "Step " + str(i+1) + ": " + response_list[i] + " <|reserved_special_token_0|>\n"

    # processed_solution = processed_solution + response_list[-1] + " The answer is: " + temp_answer + " <|reserved_special_token_0|>\n"
    processed_solution = re.sub(r"<\|end_of_text\|>", "", processed_solution)
    
    if temp_answer == answer:
        box_match = 0.5
    else:
        box_match = -0.5
    return processed_solution, box_match


def preprocess_box_responsev1(sequence, answer):
    temp_query = ""
    temp_response = ""

    temp_query = sequence.split("ASSISTANT:\n")[0]

    temp_response = sequence.split("ASSISTANT:\n")[-1]
    
    pattern = r"The final answer is: \\boxed\{(.*?)\}"
    # 使用 re.DOTALL 确保能匹配跨行文本
    match = re.search(pattern, temp_response, re.DOTALL)
    #match = re.search(pattern, temp_response)
    if match:
        temp_answer = match.group(1)
    else:
        temp_answer = "none"

    #temp_answer = temp_response.split("\n\n# Answer\n\n")[-1]
    temp_response = sequence.split("<|reserved_special_token_0|>The final answer is:")[0]

    #response_list = temp_response.split("<|reserved_special_token_0|>")

    processed_solution = temp_response + "\n\n# Answer\n\n" + temp_answer + "<|reserved_special_token_0|>"

    processed_solution = re.sub(r"<\|end_of_text\|>", "", processed_solution)
    
    if math_equal(prediction=temp_answer, reference=answer, use_timeout=True):
        box_match = 1.0
    else:
        box_match = -1.0
    
    # if temp_answer == answer:
    #     box_match = 0.5
    # else:
    #     box_match = -0.5
    return processed_solution, box_match


def preprocess_box_responsev2(sequence, answer):
    temp_query = ""
    temp_response = ""

    temp_query = sequence.split("ASSISTANT:\n")[0]

    temp_response = sequence.split("ASSISTANT:\n")[-1]
    
    pattern = r"The final answer is: \\boxed\{(.*?)\}"
    # 使用 re.DOTALL 确保能匹配跨行文本
    match = re.search(pattern, temp_response, re.DOTALL)
    #match = re.search(pattern, temp_response)
    if match:
        temp_answer = match.group(1)
    else:
        temp_answer = "none"

    #temp_answer = temp_response.split("\n\n# Answer\n\n")[-1]
    temp_response = sequence.split("<|reserved_special_token_0|>The final answer is:")[0]

    #response_list = temp_response.split("<|reserved_special_token_0|>")

    processed_solution = temp_response + "\n\n# Answer\n\n" + temp_answer + "<|reserved_special_token_0|>"

    processed_solution = re.sub(r"<\|end_of_text\|>", "", processed_solution)
    
    if math_equal(prediction=temp_answer, reference=answer, use_timeout=True):
        box_match = 0.5
    else:
        box_match = -0.5
    
    # if temp_answer == answer:
    #     box_match = 0.5
    # else:
    #     box_match = -0.5
    return processed_solution, box_match


def preprocess_box_responsev3(sequence, answer):
    temp_query = ""
    temp_response = ""

    temp_query = sequence.split("ASSISTANT:\n")[0]

    temp_response = sequence.split("**Final Answer**")[-1]
    
    pattern = r'\\boxed\{(.*?)\}\s*\\\]'

    # 使用 re.DOTALL 确保能匹配跨行文本
    match = re.search(pattern, temp_response)
    #match = re.search(pattern, temp_response)
    if match:
        temp_answer = match.group(1)
    else:
        temp_answer = "none"

    #temp_answer = temp_response.split("\n\n# Answer\n\n")[-1]
    #temp_response = sequence.split("<|reserved_special_token_0|>The final answer is:")[0]

    #response_list = temp_response.split("<|reserved_special_token_0|>")

    processed_solution = temp_response + "\n\n# Answer\n\n" + temp_answer + "<|reserved_special_token_0|>"

    processed_solution = re.sub(r"<\|end_of_text\|>", "", processed_solution)
    
    if math_equal(prediction=temp_answer, reference=answer, use_timeout=True):
        box_match = 1.0
    else:
        box_match = -1.0

    return processed_solution, box_match


def math_equal_with_timeout(prediction, reference, include_percentage=True, is_close=True, timeout_duration=10):
    """
    使用Ray的超时机制对math_equal函数进行控制
    """
    @ray.remote
    def _remote_math_equal(prediction, reference, include_percentage, is_close):
        return math_equal(prediction, reference, include_percentage, is_close, use_timeout=False)
    
    try:
        # 启动远程任务并等待结果
        future = _remote_math_equal.remote(prediction, reference, include_percentage, is_close)
        result = ray.get(future, timeout=timeout_duration)
        return result
    except (GetTimeoutError, Exception) as e:
        # 如果超时或发生其他错误，返回False
        ray.logger.info("Math Eq eval timeout.")
        return False


def preprocess_box_responsev4(sequence, answer):
    temp_query = ""
    temp_response = ""

    temp_query = sequence.split("ASSISTANT:\n")[0]

    temp_response = sequence.split("**Final Answer**")[-1]
    
    pattern = r'\\boxed\{(.*?)\}\s*\\\]'

    # 使用 re.DOTALL 确保能匹配跨行文本
    match = re.search(pattern, temp_response)
    #match = re.search(pattern, temp_response)
    if match:
        temp_answer = match.group(1)
    else:
        temp_answer = "none"

    #temp_answer = temp_response.split("\n\n# Answer\n\n")[-1]
    #temp_response = sequence.split("<|reserved_special_token_0|>The final answer is:")[0]

    #response_list = temp_response.split("<|reserved_special_token_0|>")

    processed_solution = temp_response + "\n\n# Answer\n\n" + temp_answer + "<|reserved_special_token_0|>"

    processed_solution = re.sub(r"<\|end_of_text\|>", "", processed_solution)
    
    if math_equal_with_timeout(prediction=temp_answer, reference=answer):
        box_match = 1.0
    else:
        box_match = -0.5
        
    if "**Final Answer**" not in sequence:
        box_match = -1.0
        

    return processed_solution, box_match

from openrlhf.trainer.ppo_utils.qwen_math_eval_toolkit.grader import math_equal as qwen_math_equal


def qwen_math_equal_with_timeout_ray(prediction, reference, include_percentage=True, is_close=True, timeout_duration=3):
    """
    使用Ray的超时机制对math_equal函数进行控制
    """
    @ray.remote
    def _remote_qwen_math_equal(prediction, reference, include_percentage, is_close):
        return qwen_math_equal(prediction, reference, include_percentage, is_close, timeout=False)
    
    try:
        # 启动远程任务并等待结果
        future = _remote_qwen_math_equal.remote(prediction, reference, include_percentage, is_close)
        result = ray.get(future, timeout=timeout_duration)
        return result
    except (GetTimeoutError, Exception) as e:
        # 如果超时或发生其他错误，返回False
        ray.logger.info("Math Eq eval timeout.")
        return False
    

from multiprocessing import Process, Queue


def qwen_math_equal_subprocess(prediction, reference,  timeout_seconds=10):
    def worker(q, prediction, reference):
        result = qwen_math_equal(prediction=prediction, reference=reference, timeout=False)
        q.put(result)

    q = Queue()
    p = Process(target=worker, args=(q, prediction, reference))
    p.start()
    
    # 添加超时处理
    p.join(timeout=timeout_seconds)  # 等待进程完成，最多等待 timeout_seconds 秒
    
    # 如果进程还在运行，则终止它并返回 False
    if p.is_alive():
        p.terminate()
        p.join()  # 确保进程被完全清理
        return False
        
    # 如果进程正常完成，获取结果
    try:
        return q.get_nowait()
    except:
        return False


def preprocess_box_response_for_qwen_prompt(sequence, answer, task, tokenizer, tokenizer_reward, prompt_max_len,\
    args, generate_kwargs, packed_seq_lens):
    # 从结果中提取答案
    assert("<think>" in sequence)
    if "<|im_start|>assistant\n<think>" in sequence:
        # Qwen-Instruct Template
        model_output = sequence[sequence.rfind("<|im_start|>assistant\n<think>")+len("<|im_start|>assistant\n"):]
        template = "qwen"
    
    elif ("<｜begin▁of▁sentence｜>" in sequence) and ("<｜User｜>" in sequence) and ("<｜Assistant｜><think>" in sequence):
        # DeepScaleR-1.5B-Preview/DeepSeek-R1-Distill-LLAMA-70B/Qwen-1.5B/7B Template
        model_output = sequence[sequence.rfind("<｜Assistant｜><think>")+len("<｜Assistant｜>"):]
        template = "deepseek"
    
    elif ("<|start_header_id|>" in sequence) and ("<|end_header_id|>" in sequence) and ("assistant<|end_header_id|><think>" in sequence):
        # Llama3.3-70B/llama3.1-8B Template
        model_output = sequence[sequence.rfind("assistant<|end_header_id|><think>")+len("assistant<|end_header_id|>"):]
        template = "llama3"

    elif ("[INST]" in sequence and "[/INST]<think>" in sequence):
        # llama2/mistral template
        model_output = sequence[sequence.rfind("[/INST]<think>")+len("[/INST]"):]
        template = "mistral"

    elif "Assistant:<think>" in sequence:
        # Base Template
        model_output = sequence[sequence.rfind("Assistant:<think>")+len("Assistant:"):]
        template = "base"
        
    else:
        print("[Error] Failed to locate model response header from ", json.dumps(sequence, ensure_ascii=False))
        if task.startswith("complex"):
            return "", -2.0, 0.0, 0, [], -1.0, 0.0, [], sequence, sequence, [], 0, 0
        else:
            return "", -2.0, -1.0, 0.0, 0.0, sequence, sequence, 0, True

    model_input = sequence[:sequence.find(model_output)]
    model_output = model_output.lstrip()
    # n-gram  相比于demystify long CoT的初始长度4K->770; 选择40/5.31->8
    N_gram = 8
    P_repetition = -0.1

    # repetition_penalty = get_repetition_penalty(ngram_size=N_gram, max_penalty=P_repetition, generation=model_output)
    repetition_penalty = get_repetition_penalty_tokens(ngram_size=N_gram, max_penalty=P_repetition, generation=tokenizer.encode(model_output))
    assert(repetition_penalty <= 0 and repetition_penalty >= P_repetition)
    ray.logger.info(f"[TASK={task}]: repetition_penalty={repetition_penalty}")

    stop_words = ["</s>", "<|im_end|>", "<|endoftext|>", "<｜end▁of▁sentence｜>", "<|eot_id|>"]
    for stop_word in stop_words:
        if stop_word in model_output:
            model_output = model_output.split(stop_word)[0].strip()

    # format reward
    format_reward = 1.0
    # format_correct = compute_format_reward(model_output)
    format_correct = validate_response_structure(model_output, template=template)
    format_score = format_reward if format_correct else -abs(format_reward)

    is_format_correct = 0.0
    if format_correct:
        is_format_correct = 1.0

    is_answer_superior_to_answer_woCoT = True
    # only ignore it

    # accuracy reward
    is_answer_correct = 0.0
    if task == "default":
        # accuracy reward for math problem
        # TODO: check the data_name, hard code here 
        extract_answer = qwen_extract_answer(model_output, data_name="math")
        if qwen_math_equal_subprocess(prediction=extract_answer, reference=answer):
            answer_score = 2.0
            is_answer_correct = 1.0
        else:
            answer_score = -1.5
        if "boxed" not in model_output:
            answer_score = -2.0
        repetition_penalty = 0
    
    elif task.startswith("kk"):
        # calculate for logic kk problem scale = [-2, 2]
        ignore_format = False
        if task.endswith("woFormat"):
            format_score = 1.0
            ignore_format = True
        answer_score, is_answer_correct = compute_kk_score(model_output, answer, template=template, ignore_format=ignore_format)
        repetition_penalty = 0
    
    elif task.startswith("ifeval"):
        answer_json = json.loads(answer)
        ignore_format = False
        if task.endswith("woFormat"):
            format_score = 1.0
            ignore_format = True
        answer_score, is_answer_correct, is_answer_superior_to_answer_woCoT = compute_ifeval_score(model_output, answer, answer_json,\
            template=template, ignore_format=ignore_format)
        assert(answer_score >= -2 and answer_score <= 2)
        # if-eval需要考虑repetition penalty
        # answer_score = max(-2.0, answer_score+repetition_penalty)

    elif task.startswith("complex"):
        answer_json = json.loads(answer)  # here are all the scoring questions
        ignore_format = False
        if task.endswith("woFormat"):
            format_score = 1.0
            ignore_format = True
        if "scoring_sum_qwen2.5-7B-Instruct" in answer_json:
            # TODO: replace the off-line scoring with the on-line scoring
            reference_scoring_woCoT = answer_json["scoring_sum_qwen2.5-7B-Instruct"]
        else:
            # if not existing simply ignore it
            reference_scoring_woCoT = 0

        if (template == "qwen" and "<|im_start|>user" in model_input) or \
            (template == "deepseek" and "<｜User｜>" in model_input) or \
                (template == "llama3" and "user<|end_header_id|>" in model_input) or \
                    (template == "mistral" and "[INST]" in model_input) or \
                        (template == "base" and "User:" in model_input):
            answer_score, is_answer_correct, wrapping_content_list_all, wrapping_pairs_list_all,\
                num_actions, scoring_questions = prepare_complex_judgement(model_input, model_output, answer_json, tokenizer_reward,\
                    ignore_format=ignore_format, template=template)
            # 注意这里因为借助了-100来判断是否需要做推理(reward model judgement) 所以不能直接加repetition penalty
            # 只这种情况下才需要传回reference_scoring_woCoT
            return "", float(answer_score), float(is_answer_correct), num_actions, wrapping_content_list_all, float(format_score), float(is_format_correct), wrapping_pairs_list_all, model_input, model_output, scoring_questions, repetition_penalty, reference_scoring_woCoT
        else:
            return "", -2.0, 0.0, 0, [], -1.0, 0.0, [], sequence, sequence, [], 0, 0

    else:
        raise NotImplementedError("Not Found Available Tools")
    return "", float(answer_score), float(format_score), float(is_format_correct), float(is_answer_correct), model_input, model_output, repetition_penalty, is_answer_superior_to_answer_woCoT


def compute_format_reward(content):
    """Reward function that checks if the completion has a specific format."""
    # strict matching
    # pattern = r"^<think>.*?</think><answer>.*?</answer>$"
    pattern = r"<think>.*?</think><answer>.*?</answer>"
    match = re.match(pattern, content)

    pattern2 = r"<think>.*?</think>\n<answer>.*?</answer>"
    match2 = re.match(pattern2, content)

    if match or match2:
        return True
    else:
        return False



def preprocess_orm_reward(queries, tokenizer, **generate_kwargs):
    input_ids = []
    input_masks = []
    for input_for_prm in queries:
        input_token = tokenizer(
            input_for_prm,
            max_length=generate_kwargs["max_new_tokens"],
            padding=False,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        for input_id, input_mask in zip(input_token["input_ids"], input_token["attention_mask"]):
            input_ids.append(input_id)
            input_masks.append(input_mask)

    padding_side = "right"
    input_ids = zero_pad_sequences(input_ids, side=padding_side, value=tokenizer.pad_token_id)
    input_masks = zero_pad_sequences(input_masks, side=padding_side)

    return input_ids, input_masks


def preprocess_prm_reward(queries, tokenizer, prompt_max_len, **generate_kwargs):
    input_ids = []
    input_masks = []
    for input_for_prm in queries:
        input_token = tokenizer(
            input_for_prm,
            max_length=generate_kwargs["max_new_tokens"] + prompt_max_len,
            padding=False,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        for input_id, input_mask in zip(input_token["input_ids"], input_token["attention_mask"]):
            input_ids.append(input_id)
            input_masks.append(input_mask)

    padding_side = "right"
    input_ids = zero_pad_sequences(input_ids, side=padding_side, value=tokenizer.pad_token_id)
    input_masks = zero_pad_sequences(input_masks, side=padding_side)

    return input_ids, input_masks


def zero_pad_sequences(sequences, side: str = "left", value=0):
    assert side in ("left", "right")
    max_len = max(seq.size(-1) for seq in sequences)
    padded_sequences = []
    for seq in sequences:
        pad_len = max_len - seq.size(-1)
        padding = (pad_len, 0) if side == "left" else (0, pad_len)
        padded_sequences.append(F.pad(seq, padding, value=value))
    return torch.stack(padded_sequences, dim=0)


def to(tensor: Union[torch.Tensor, list[torch.Tensor]], device):

    if isinstance(tensor, list):
        return [to(t, device) for t in tensor]
    
    return tensor.to(device) if isinstance(tensor, torch.Tensor) else tensor


def pin_memory(tensor: Union[torch.Tensor, list[torch.Tensor]]):

    if isinstance(tensor, list):
        return [pin_memory(t) for t in tensor]
    
    return tensor.pin_memory() if isinstance(tensor, torch.Tensor) else tensor


@dataclass
class Experience:
    """Experience is a batch of data.
    These data should have the the sequence length and number of actions.
    Left padding for sequences is applied.

    Shapes of each tensor:
    sequences: (B, S)
    action_log_probs: (B, A)
    values: (B, A)
    returns: (B, A)
    advantages: (B, A)
    attention_mask: (B, S)
    action_mask: (B, A)
    kl: (B, A)
    prompts: (B)

    "A" is the number of actions.
    """

    sequences: torch.Tensor
    action_log_probs: torch.Tensor
    base_action_log_probs: torch.Tensor
    values: torch.Tensor
    returns: Optional[torch.Tensor]
    advantages: Optional[torch.Tensor]
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    info: Optional[dict]
    kl: Optional[torch.Tensor] = None
    enforce_superior_final: Optional[torch.Tensor] = None
    r_acc_all: Optional[torch.Tensor] = None
    prompts: Optional[List] = None

    @torch.no_grad()
    def to_device(self, device: torch.device):
        self.sequences = to(self.sequences, device)
        self.action_log_probs = to(self.action_log_probs, device)
        self.base_action_log_probs = to(self.base_action_log_probs, device)
        self.returns = to(self.returns, device)
        self.advantages = to(self.advantages, device)
        self.values = to(self.values, device)
        self.attention_mask = to(self.attention_mask, device)
        self.action_mask = to(self.action_mask, device)
        self.kl = to(self.kl, device)
        self.info = {key: to(value, device) for key, value in self.info.items()}
        self.enforce_superior_final = to(self.enforce_superior_final, device)
        self.r_acc_all = to(self.r_acc_all, device)
        return self

    def pin_memory(self):
        self.sequences = pin_memory(self.sequences)
        self.action_log_probs = pin_memory(self.action_log_probs)
        self.base_action_log_probs = pin_memory(self.base_action_log_probs)
        self.returns = pin_memory(self.returns)
        self.advantages = pin_memory(self.advantages)
        self.values = pin_memory(self.values)
        self.attention_mask = pin_memory(self.attention_mask)
        self.action_mask = pin_memory(self.action_mask)
        self.kl = pin_memory(self.kl)
        self.info = {key: pin_memory(value) for key, value in self.info.items()}
        self.enforce_superior_final = pin_memory(self.enforce_superior_final)
        self.r_acc_all = pin_memory(self.r_acc_all)
        self.prompts = pin_memory(self.prompts)
        return self


@dataclass
class Samples:
    """Samples is a batch of data.
    There can be 2 formats to store the samples, batched or packed.
    The batched format means padding is applied to the sequences, while the packed format
    will concatenate the prompt and response without padding.

    Shapes of each tensor, when 2 shapes are shown, the first one is for batched format
        and the second one is for packed format:
    sequences: (B, S) or (1, total_length), the tokens of both prompt and response.
    attention_mask: (B, S) or (1, total_length), the attention mask for sequences.
    action_mask: (B, A) or None, the action (response) mask to show which part of the
        sequence is the response. When the samples are packed, this is None.
    num_actions: int or (B,), the number of actions (tokens) in the response.
        When the samples are not packed, we will use action_mask, so this is an int to
        show the size of action_mask. Otherwise, this is a tensor to show the number of
        actions for each sample.
    packed_seq_lens: None or (B,), the length of each sample in the packed samples.
    response_length: (B,), the number of tokens in the response.
    total_length: (B,), the total number of tokens in the sequences.
    prompts: the prompts used to generate responses
    """

    sequences: torch.Tensor
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    num_actions: Union[int, torch.Tensor]
    packed_seq_lens: Optional[torch.Tensor]
    response_length: torch.Tensor
    total_length: torch.Tensor
    prompts: list[str]
    answers: list[str]
    tasks: list[str]
    generate_kwargs: dict


@dataclass
class SamplesBOX:
    """Samples is a batch of data.
    There can be 2 formats to store the samples, batched or packed.
    The batched format means padding is applied to the sequences, while the packed format
    will concatenate the prompt and response without padding.

    Shapes of each tensor, when 2 shapes are shown, the first one is for batched format
        and the second one is for packed format:
    sequences: (B, S) or (1, total_length), the tokens of both prompt and response.
    attention_mask: (B, S) or (1, total_length), the attention mask for sequences.
    action_mask: (B, A) or None, the action (response) mask to show which part of the
        sequence is the response. When the samples are packed, this is None.
    num_actions: int or (B,), the number of actions (tokens) in the response.
        When the samples are not packed, we will use action_mask, so this is an int to
        show the size of action_mask. Otherwise, this is a tensor to show the number of
        actions for each sample.
    packed_seq_lens: None or (B,), the length of each sample in the packed samples.
    response_length: (B,), the number of tokens in the response.
    total_length: (B,), the total number of tokens in the sequences.
    """

    sequences: torch.Tensor
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    num_actions: Union[int, torch.Tensor]
    packed_seq_lens: Optional[torch.Tensor]
    response_length: torch.Tensor
    total_length: torch.Tensor
    prompts: list[str]
    answers: list[str]
    tasks: list[str]
    generate_kwargs: dict


class NaiveExperienceMaker(ABC):
    """
    Naive experience maker.
    """

    def __init__(
        self,
        actor: Actor,
        critic: nn.Module,
        reward_model: nn.Module,
        initial_model: Actor,
        tokenizer,
        prompt_max_len: int,
        kl_controller,
        strategy=None,
        remote_rm_url: list[str] = None,
        reward_fn=None,
    ) -> None:
        super().__init__()
        self.actor = actor
        self.critic = critic
        self.reward_model = reward_model
        self.remote_rm_url = remote_rm_url
        self.initial_model = initial_model
        self.tokenizer = tokenizer
        self.prompt_max_len = prompt_max_len
        self.kl_ctl = kl_controller
        self.strategy = strategy
        self.reward_fn = reward_fn
        self.perf_stats = None
        self.advantage_estimator = strategy.args.advantage_estimator

        # custom reward func for reinforced finetuning
        self.custom_reward_func = None
        if remote_rm_url and remote_rm_url[0].endswith(".py"):
            print(f"Loading custom `reward_func(queries, prompts, labels)` from {remote_rm_url[0]}")
            import importlib.util

            spec = importlib.util.spec_from_file_location("reward_func", remote_rm_url[0])
            reward_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(reward_module)
            self.custom_reward_func = reward_module.reward_func

    # tokenizer
    def tokenize_fn(self, texts, max_length, padding=True, device=None):
        if not padding:
            # when padding is False, return tokenized texts as list
            return self.tokenizer(
                texts,
                add_special_tokens=False,
                max_length=max_length,
                truncation=True,
            )
        batch = self.tokenizer(
            texts,
            return_tensors="pt",
            add_special_tokens=False,
            max_length=max_length,
            padding=True,
            truncation=True,
        )
        return {k: v.to(device) for k, v in batch.items()}

    @torch.no_grad()
    def make_experience_list(
        self, all_prompts: Union[str, List[str]], all_labels, all_tasks, **generate_kwargs
    ) -> List[Experience]:
        """
        Make a list of experience with the micro_rollout_batch_size.

        This method will first calculate the response sequences and rewards for the given prompts.
        Then, if we need certain processing for the rewards or do certain filtering, we can process the rollout as a whole.
        After that, we will calculate the advantages and returns for each experience.
        """
        args = self.strategy.args
        samples_list = self.generate_samples(all_prompts, all_labels, all_tasks, **generate_kwargs)
        torch.distributed.barrier()

        experiences = []
        for samples in tqdm(
            samples_list,
            desc="make_experience",
            disable=not self.strategy.is_rank_0(),
        ):
            experiences.append(self.make_experience(samples).to_device("cpu"))

        experiences, rewards = self.process_experiences(experiences)
        torch.distributed.barrier()
        # calculate return and advantages
        for experience, reward in zip(experiences, rewards):
            experience = experience.to_device("cuda")
            reward = reward.to(device="cuda")
            num_actions = experience.info["num_actions"]
            reward = compute_reward(
                reward,
                self.kl_ctl.value,
                experience.kl,
                action_mask=experience.action_mask,
                num_actions=num_actions,
                reward_clip_range=args.reward_clip_range,
            )

            if self.advantage_estimator == "gae":
                experience.advantages, experience.returns = self.get_advantages_and_returns(
                    experience.values,
                    reward,
                    experience.action_mask,
                    generate_kwargs["gamma"],
                    generate_kwargs["lambd"],
                )
            elif self.advantage_estimator in ["reinforce", "rloo", "reinforce_baseline", "group_norm"]:
                experience.returns = self.get_cumulative_returns(
                    reward,
                    experience.action_mask,
                    generate_kwargs["gamma"],
                )
                experience.advantages = deepcopy(experience.returns)
            else:
                raise Exception(f"Unkown advantage_estimator {self.advantage_estimator}")

            # calculate the return info.
            if not getattr(self, "packing_samples", False):
                return_sums = reward.sum(dim=-1)
            else:
                return_sums = torch.tensor(
                    [each_reward.sum() for each_reward in reward], device=torch.cuda.current_device()
                )
            experience.info["return"] = return_sums
            # remove unnecessary info
            experience.kl = None
            del experience.info["num_actions"]
            experience.to_device("cpu")
        return experiences

    @torch.no_grad()
    def generate_samples(self, all_prompts: List[str], all_labels, all_tasks, **generate_kwargs) -> List[Samples]:
        """
        Generate samples and return in batches.
        """
        assert not getattr(self, "packing_samples", False)
        args = self.strategy.args
        self.actor.eval()
        # sample multiple response
        all_prompts = sum([[prompt] * args.n_samples_per_prompt for prompt in all_prompts], [])
        all_labels = sum([[label] * args.n_samples_per_prompt for label in all_labels], [])
        all_tasks = sum([[task] * args.n_samples_per_prompt for task in all_tasks], [])

        samples_list = []
        for i in range(0, len(all_prompts), args.micro_rollout_batch_size):
            prompts = all_prompts[i : i + args.micro_rollout_batch_size]
            labels = all_labels[i : i + args.micro_rollout_batch_size]
            tasks = all_tasks[i : i + args.micro_rollout_batch_size]
            inputs = self.tokenize_fn(prompts, self.prompt_max_len, device="cuda")
            sequences, attention_mask, action_mask = self.actor.generate(**inputs, **generate_kwargs)
            samples = Samples(
                sequences=sequences,
                attention_mask=attention_mask,
                action_mask=action_mask,
                num_actions=action_mask.size(1),
                packed_seq_lens=None,
                response_length=action_mask.float().sum(dim=-1),
                total_length=attention_mask.float().sum(dim=-1),
                prompts=prompts,
                labels=labels,
                tasks=tasks,
                generate_kwargs=generate_kwargs,
            )
            samples_list.append(samples)
        return samples_list

    @torch.no_grad()
    def make_experience(self, samples: Samples) -> Experience:
        """
        Turn samples into experience by calculating logprobs, values, rewards, and kl divergence.
        """
        self.actor.eval()
        if self.initial_model is not None:
            self.initial_model.eval()
        if self.reward_model is not None:
            self.reward_model.eval()
        if self.critic is not None:
            self.critic.eval()

        # extract values from samples
        sequences = samples.sequences
        attention_mask = samples.attention_mask
        action_mask = samples.action_mask
        num_actions = samples.num_actions
        tasks = samples.tasks
        # log probs
        action_log_probs = self.actor(sequences, num_actions, attention_mask)

        # init log probs
        if self.initial_model is not None:
            base_action_log_probs = self.initial_model(sequences, num_actions, attention_mask)
        else:
            base_action_log_probs = None

        # values
        if self.critic is not None:
            value = self.critic(sequences, num_actions, attention_mask)
        else:
            value = None

        # rewards
        if self.remote_rm_url is not None:
            # remote RM
            queries = self.tokenizer.batch_decode(sequences.cpu(), skip_special_tokens=False)
            if self.custom_reward_func:
                r = self.custom_reward_func(queries, samples.prompts, samples.labels).to(
                    device=action_log_probs.device
                )
            else:
                r = remote_rm_fn(
                    self.remote_rm_url, queries=queries, prompts=samples.prompts, labels=samples.labels
                ).to(device=action_log_probs.device)
        else:
            # local RM
            r = self.reward_model(sequences, attention_mask)

        if (self.initial_model is not None) and (not self.strategy.args.use_kl_loss):
            kl = compute_approx_kl(
                action_log_probs,
                base_action_log_probs,
                action_mask=action_mask,
                use_kl_estimator_k3=self.strategy.args.use_kl_estimator_k3,
            )
        else:
            kl = torch.zeros_like(action_log_probs, dtype=action_log_probs.dtype, device=action_log_probs.device)

        info = {
            "kl": masked_mean(kl, action_mask, dim=-1),
            "reward": r,
            "response_length": samples.response_length,
            "total_length": samples.total_length,
            "num_actions": num_actions,
        }
        # reset model state
        self.actor.train()
        if self.critic is not None:
            self.critic.train()

        return Experience(
            sequences,
            action_log_probs,
            base_action_log_probs,
            value,
            None,
            None,
            attention_mask,
            action_mask,
            info,
            kl,
        )

    @torch.no_grad()
    def process_experiences(self, experiences: List[Experience]) -> Tuple[List[Experience], List[torch.Tensor]]:
        """
        Process experiences, this can be used to filter out some experiences or do some processing on the rewards.

        Output:
        - experiences: List of Experience
        - rewards: List of rewards
        """
        args = self.strategy.args
        # reward shaping for rloo and reinforce_baseline
        if args.advantage_estimator == "rloo":
            rewards = torch.cat([experience.info["reward"] for experience in experiences])
            rewards = rewards.reshape(-1, args.n_samples_per_prompt).to(device="cuda")
            baseline = (rewards.sum(-1, keepdim=True) - rewards) / (args.n_samples_per_prompt - 1)
            rewards = rewards - baseline
            rewards = rewards.flatten().to(device="cpu").chunk(len(experiences))
            return experiences, rewards
        elif args.advantage_estimator == "reinforce_baseline":
            # REINFORCE++-baseline removed the / std and K3 kl loss in GRPO.
            # `/ std` is not needed in RL variance reduction theory, and `k3 KL` has a larger variance than `k1 KL` under a categorical distribution.
            rewards = torch.cat([experience.info["reward"] for experience in experiences])
            rewards = rewards.reshape(-1, args.n_samples_per_prompt).to(device="cuda")
            rewards = rewards - rewards.mean(-1, keepdim=True)
            rewards = rewards.reshape(-1).to(device="cpu").chunk(len(experiences))
            return experiences, rewards
        elif args.advantage_estimator == "group_norm":
            rewards = torch.cat([experience.info["reward"] for experience in experiences])
            rewards = rewards.reshape(-1, args.n_samples_per_prompt).to(device="cuda")
            rewards = (rewards - rewards.mean(-1, keepdim=True)) / (rewards.std(-1, keepdim=True) + 1e-9)
            rewards = rewards.reshape(-1).to(device="cpu").chunk(len(experiences))
            return experiences, rewards
        # default rewards
        return experiences, [experience.info["reward"] for experience in experiences]

    @torch.no_grad()
    def get_advantages_and_returns(
        self,
        values: torch.Tensor,
        rewards: torch.Tensor,
        action_mask: torch.Tensor,
        gamma: float,
        lambd: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Function that computes advantages and returns from rewards and values.
        Calculated as in the original PPO paper: https://arxiv.org/abs/1707.06347
        Note that rewards may include a KL divergence loss term.

        Advantages looks like this:
        Adv1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
              - V1 + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Returns looks like this:
        Ret1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
                   + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Input:
        - values: Tensor of shape (batch_size, response_size)
        - rewards: Tensor of shape (batch_size, response_size)

        Output:
        - advantages: Tensor of shape (batch_size, response_size)
        - returns: Tensor of shape (batch_size, response_size)
        """
        if isinstance(values, list):
            # packing samples
            # TODO: this is slow...
            advantages = []
            returns = []
            for v, r in zip(values, rewards):
                adv, ret = self.get_advantages_and_returns(v.unsqueeze(0), r.unsqueeze(0), action_mask, gamma, lambd)
                advantages.append(adv.squeeze(0))
                returns.append(ret.squeeze(0))
            return advantages, returns

        lastgaelam = 0
        advantages_reversed = []
        response_length = rewards.size(1)

        # Mask invalid responses
        if action_mask is not None:
            values = action_mask * values
            rewards = action_mask * rewards

        for t in reversed(range(response_length)):
            nextvalues = values[:, t + 1] if t < response_length - 1 else 0.0
            delta = rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lambd * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values
        return advantages.detach(), returns

    @torch.no_grad()
    def get_cumulative_returns(
        self,
        rewards: torch.Tensor,
        action_mask: torch.Tensor,
        gamma: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Function that computes advantages and returns from rewards using REINFORCE.
        REINFORCE uses cumulative returns without the GAE (Generalized Advantage Estimation).

        Input:
        - rewards: Tensor of shape (batch_size, response_size)
        - action_mask: Tensor of shape (batch_size, response_size), binary mask
        - gamma: discount factor

        Output:
        - returns: Tensor of shape (batch_size, response_size)
        """

        if isinstance(rewards, list):
            # packing samples
            # TODO: this is slow...
            returns = []
            for r in rewards:
                ret = self.get_cumulative_returns(r.unsqueeze(0), action_mask, gamma)
                returns.append(ret.squeeze(0))
            return returns

        response_length = rewards.size(1)
        returns = torch.zeros_like(rewards)
        cumulative_return = torch.zeros(rewards.size(0), device=rewards.device)

        # Mask invalid responses if action_mask is provided
        if action_mask is not None:
            rewards = action_mask * rewards

        # Calculate returns by accumulating discounted rewards
        for t in reversed(range(response_length)):
            cumulative_return = rewards[:, t] + gamma * cumulative_return
            returns[:, t] = cumulative_return

        return returns


class NaiveExperienceMakerBOX(ABC):
    """
    Naive experience maker.
    """

    def __init__(
        self,
        actor: Actor,
        critic: nn.Module,
        reward_model: nn.Module,
        initial_model: Actor,
        tokenizer,
        prompt_max_len: int,
        kl_controller,
        strategy=None,
        remote_rm_url: list[str] = None,
        reward_fn=None,
    ) -> None:
        super().__init__()
        self.actor = actor
        self.critic = critic
        self.reward_model = reward_model
        self.remote_rm_url = remote_rm_url
        self.initial_model = initial_model
        self.tokenizer = tokenizer
        self.prompt_max_len = prompt_max_len
        self.kl_ctl = kl_controller
        self.strategy = strategy
        self.reward_fn = reward_fn
        self.perf_stats = None
        self.advantage_estimator = strategy.args.advantage_estimator

        # custom reward func for reinforced finetuning
        self.custom_reward_func = None
        if remote_rm_url and remote_rm_url[0].endswith(".py"):
            print(f"Loading custom `reward_func(queries, prompts, labels)` from {remote_rm_url[0]}")
            import importlib.util

            spec = importlib.util.spec_from_file_location("reward_func", remote_rm_url[0])
            reward_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(reward_module)
            self.custom_reward_func = reward_module.reward_func

    # tokenizer
    def tokenize_fn(self, texts, max_length, padding=True, device=None):
        if not padding:
            # when padding is False, return tokenized texts as list
            return self.tokenizer(
                texts,
                add_special_tokens=False,
                max_length=max_length,
                truncation=True,
            )
        batch = self.tokenizer(
            texts,
            return_tensors="pt",
            add_special_tokens=False,
            max_length=max_length,
            padding=True,
            truncation=True,
        )
        return {k: v.to(device) for k, v in batch.items()}

    @torch.no_grad()
    def make_experience_list(
        self, all_prompts: Union[str, List[str]], all_labels, all_tasks, **generate_kwargs
    ) -> List[Experience]:
        """
        Make a list of experience with the micro_rollout_batch_size.

        This method will first calculate the response sequences and rewards for the given prompts.
        Then, if we need certain processing for the rewards or do certain filtering, we can process the rollout as a whole.
        After that, we will calculate the advantages and returns for each experience.
        """
        args = self.strategy.args
        experiences = []
        samples_list = self.generate_samples(all_prompts, all_labels, all_tasks, **generate_kwargs)
        torch.distributed.barrier()

        for idx, samples in enumerate(tqdm(
            samples_list,
            desc=f"make_experience",
            disable=not self.strategy.is_rank_0(),
            )):
            experiences.append(self.make_experience(samples).to_device("cpu"))

        experiences, rewards = self.process_experiences(experiences)
        torch.distributed.barrier()
        # calculate return and advantages
        for experience, reward in zip(experiences, rewards):
            experience = experience.to_device("cuda")
            reward = reward.to(device="cuda")
            num_actions = experience.info["num_actions"]
            reward = compute_reward(
                reward,
                self.kl_ctl.value,
                experience.kl,
                action_mask=experience.action_mask,
                num_actions=num_actions,
                reward_clip_range=args.reward_clip_range,
            )

            if self.advantage_estimator == "gae":
                experience.advantages, experience.returns = self.get_advantages_and_returns(
                    experience.values,
                    reward,
                    experience.action_mask,
                    generate_kwargs["gamma"],
                    generate_kwargs["lambd"],
                )
            elif self.advantage_estimator in ["reinforce", "rloo", "reinforce_baseline", "group_norm"]:
                experience.returns = self.get_cumulative_returns(
                    reward,
                    experience.action_mask,
                    generate_kwargs["gamma"],
                )
                experience.advantages = deepcopy(experience.returns)
            else:
                raise Exception(f"Unkown advantage_estimator {self.advantage_estimator}")

            # calculate the return info.
            if not getattr(self, "packing_samples", False):
                return_sums = reward.sum(dim=-1)
            else:
                return_sums = torch.tensor(
                    [each_reward.sum() for each_reward in reward], device=torch.cuda.current_device()
                )
            experience.info["return"] = return_sums
            # remove unnecessary info
            experience.kl = None
            del experience.info["num_actions"]
            experience.to_device("cpu")
        return experiences

    @torch.no_grad()
    def generate_samples(self, all_prompts: List[str], all_labels, all_tasks, **generate_kwargs) -> List[Samples]:
        """
        Generate samples and return in batches.
        """
        assert not getattr(self, "packing_samples", False)
        args = self.strategy.args
        self.actor.eval()
        # sample multiple response
        all_prompts = sum([[prompt] * args.n_samples_per_prompt for prompt in all_prompts], [])
        all_labels = sum([[label] * args.n_samples_per_prompt for label in all_labels], [])
        all_tasks = sum([[task] * args.n_samples_per_prompt for task in all_tasks], [])
        samples_list = []
        #self.strategy.print(f"generate samples!!!")
        for i in range(0, len(all_prompts), args.micro_rollout_batch_size):
            prompts = all_prompts[i : i + args.micro_rollout_batch_size]
            labels = all_labels[i : i + args.micro_rollout_batch_size]
            tasks = all_tasks[i : i + args.micro_rollout_batch_size]
            inputs = self.tokenize_fn(prompts, self.prompt_max_len, device="cuda")
            #self.strategy.print(f"generating!!!")
            sequences, attention_mask, action_mask = self.actor.generate(**inputs, **generate_kwargs)
            #self.strategy.print(f"generating!!!", sequences)
            samples = SamplesBOX(
                sequences=sequences,
                attention_mask=attention_mask,
                action_mask=action_mask,
                num_actions=action_mask.size(1),
                packed_seq_lens=None,
                response_length=action_mask.float().sum(dim=-1),
                total_length=attention_mask.float().sum(dim=-1),
                prompts=prompts,
                answers=labels,
                tasks=tasks,
                generate_kwargs=generate_kwargs,
            )
            samples_list.append(samples)

            #print("sequences", samples.sequences)
            
            #print("attention_mask", samples.attention_mask)


        return samples_list

    @torch.no_grad()
    def make_experience(self, samples: Samples, **generate_kwargs) -> Experience:
        """
        Turn samples into experience by calculating logprobs, values, rewards, and kl divergence.
        """
        if self.initial_model is not None:
            self.initial_model.eval()
        self.initial_model.eval()
        if self.reward_model is not None:
            self.reward_model.eval()
        if self.critic is not None:
            self.critic.eval()

        # extract values from samples
        sequences = samples.sequences
        attention_mask = samples.attention_mask
        action_mask = samples.action_mask
        num_actions = samples.num_actions
        answers = samples.answers
        tasks = samples.tasks
        # log probs
        #print("sequences.shape", sequences.shape)
        #print("sequences", sequences[0])
        #print("num_actions", num_actions)
        #print("attention_mask", attention_mask.shape)
        #print("action_mask", action_mask.shape)
        action_log_probs = self.actor(sequences, num_actions, attention_mask)
        
        #print("action_log_probs", action_log_probs.shape)

        # init log probs
        if self.initial_model is not None:
            base_action_log_probs = self.initial_model(sequences, num_actions, attention_mask)
        else:
            base_action_log_probs = None

        # values
        if self.critic is not None:
            value = self.critic(sequences, num_actions, attention_mask)
        else:
            value = None

        # rewards
        if self.remote_rm_url is not None:
            # remote RM
            queries = self.tokenizer.batch_decode(sequences.cpu(), skip_special_tokens=False)
            if self.custom_reward_func:
                r = self.custom_reward_func(queries, samples.prompts, samples.labels).to(
                    device=action_log_probs.device
                )
        else:
            # local RM
            #print("sequences", sequences[0])
            queries = self.tokenizer.batch_decode(sequences.cpu(), skip_special_tokens=False)
            #print("queries", queries[0])
            # 监测不包含 \n\n# Answer\n\n 的 queries
            #print("queries", queries[0])
            # 定义正则模式
            #pattern = r"The final answer is: \\\\boxed\{(.*?)\}"
            pattern = r"The final answer is: \\boxed\{(.*?)\}"
            # 找到没有匹配到内容的索引
            missing_answer_indices = [
            i for i, query in enumerate(queries) if not re.search(pattern, query, re.DOTALL)
            ]
            processed_queries = []
            box_match_list = []
            for query, answer in zip(queries, answers):
                query, box_match = preprocess_orm800k_box_responsev1(query, answer)
                processed_queries.append(query)
                box_match_list.append(box_match)
                
            queries = processed_queries
            reward_sequences, reward_attention_mask = preprocess_prm_reward(queries, self.tokenizer, self.prompt_max_len, **generate_kwargs)
            candidate_tokens = [128003, 128004]
            logits = self.reward_model(reward_sequences.to(device=action_log_probs.device), attention_mask=reward_attention_mask.to(device=action_log_probs.device), return_output=True).logits[:, :, candidate_tokens]
            scores = logits.softmax(dim=-1)[:, :, 0]
            step_scores = []
            step_tag = '<|reserved_special_token_0|>'
            step_tag_id = self.tokenizer.encode(f"{step_tag}")[-1]

            for index, score in enumerate(scores):
                if index in missing_answer_indices:
                # 如果 query 不包含 \n\n# Answer\n\n，设置 step_score 为 0.0
                    step_scores.append(torch.tensor(0.0, device=action_log_probs.device))
                else:
                    # 否则正常计算 step_score
                    step_scores.append(score[reward_sequences[index] == step_tag_id])
                    
            reward_indexs = []
            for seq in sequences:
                indices = (seq == step_tag_id).nonzero(as_tuple=True)[0]
                if len(indices) > 0:
                    #print("neg num actions", -num_actions)
                    reverse_indices = [- (len(seq) - idx.item()) for idx in indices]
                    #print("reverse_indices", reverse_indices)
                    
                else:
                    reverse_indices = [-1]
                    
                reward_indexs.append(torch.tensor(reverse_indices, device=action_log_probs.device))
                
                
            # 使用负索引提取128002的位置
            extracted_elements = []
            for seq, negative_indices in zip(sequences, reward_indexs):
                elements = [seq[idx].item() for idx in negative_indices]
                extracted_elements.append(elements)

            # 提取每个张量的最小值
            #min_values = [step_score.min() for step_score in step_scores]

            # 提取每个张量的最小值，若为空则设置为0.0
            min_values = [step_score.min()-0.5 if step_score.numel() > 0 else torch.tensor(-0.5, device=action_log_probs.device) for step_score in step_scores]
            # 将最小值重新合并为一个新的张量
            min_values_tensor = torch.tensor(min_values, device=action_log_probs.device)

            r = min_values_tensor.to(device=action_log_probs.device)
            
            step_scores = [step_score - 0.5 if step_score.numel() > 0 else torch.tensor(-0.5, device=action_log_probs.device) for step_score in step_scores]
            
            # 将 math_box_list 转换为张量，并移动到与 step_scores 相同的设备
            math_box_tensor = torch.tensor(box_match_list, device=r.device)
            # print("r", r)
            
            # print("step_scores", step_scores)
            if len(math_box_tensor) == len(step_scores):
                
                r = r + math_box_tensor
                for i in range(len(math_box_tensor)):
                    step_score = step_scores[i]
                    add_box = math_box_tensor[i]
                    
                    if step_score.dim() == 0:
                        step_scores[i] = step_score + add_box
                    else:
                        step_scores[i][-1] += add_box

        if (self.initial_model is not None) and (not self.strategy.args.use_kl_loss):   
            kl = compute_approx_kl(
                action_log_probs,
                base_action_log_probs,
                action_mask=action_mask,
                use_kl_estimator_k3=self.strategy.args.use_kl_estimator_k3,
            )
            #print("kl.shape", kl.shape)
        else:
            kl = torch.zeros_like(action_log_probs, dtype=action_log_probs.dtype, device=action_log_probs.device)

        for i, tensor in enumerate(reward_indexs):
            if not torch.equal(tensor, torch.tensor([-1], device=action_log_probs.device)):
                # Remove the second last element
                reward_indexs[i] = torch.cat((tensor[:-2], tensor[-1:]))
            
        #print(masked_mean(kl, action_mask, dim=-1).shape)
        info = {
            "kl": masked_mean(kl, action_mask, dim=-1),
            "reward": r,
            "response_length": samples.response_length,
            "total_length": samples.total_length,
            "num_actions": num_actions,
            #"reward_indexs": reward_indexs,
            #"step_rewards": step_scores,
        }
        # reset model state
        self.actor.train()
        if self.critic is not None:
            self.critic.train()
            
        #print("pre_pre_action_mask", action_mask.size(1))

        return Experience(
            sequences,
            action_log_probs,
            base_action_log_probs,
            value,
            None,
            None,
            attention_mask,
            action_mask,
            info,
            kl,
        ), reward_indexs, step_scores

    @torch.no_grad()
    def process_experiences(self, experiences: List[Experience]) -> Tuple[List[Experience], List[torch.Tensor]]:
        """
        Process experiences, this can be used to filter out some experiences or do some processing on the rewards.

        Output:
        - experiences: List of Experience
        - rewards: List of rewards
        """
        args = self.strategy.args
        # reward shaping for rloo and reinforce_baseline
        if args.advantage_estimator == "rloo":
            rewards = torch.cat([experience.info["reward"] for experience in experiences])
            rewards = rewards.reshape(-1, args.n_samples_per_prompt).to(device="cuda")
            baseline = (rewards.sum(-1, keepdim=True) - rewards) / (args.n_samples_per_prompt - 1)
            rewards = rewards - baseline
            rewards = rewards.flatten().to(device="cpu").chunk(len(experiences))
            return experiences, rewards
        elif args.advantage_estimator == "reinforce_baseline":
            # REINFORCE++-baseline removed the / std and K3 kl loss in GRPO.
            # `/ std` is not needed in RL variance reduction theory, and `k3 KL` has a larger variance than `k1 KL` under a categorical distribution.
            rewards = torch.cat([experience.info["reward"] for experience in experiences])
            rewards = rewards.reshape(-1, args.n_samples_per_prompt).to(device="cuda")
            rewards = rewards - rewards.mean(-1, keepdim=True)
            rewards = rewards.reshape(-1).to(device="cpu").chunk(len(experiences))
            return experiences, rewards
        elif args.advantage_estimator == "group_norm":
            rewards = torch.cat([experience.info["reward"] for experience in experiences])
            rewards = rewards.reshape(-1, args.n_samples_per_prompt).to(device="cuda")
            rewards = (rewards - rewards.mean(-1, keepdim=True)) / (rewards.std(-1, keepdim=True) + 1e-9)
            rewards = rewards.reshape(-1).to(device="cpu").chunk(len(experiences))
            return experiences, rewards
        # default rewards
        return experiences, [experience.info["reward"] for experience in experiences]

    @torch.no_grad()
    def get_advantages_and_returns(
        self,
        values: torch.Tensor,
        rewards: torch.Tensor,
        action_mask: torch.Tensor,
        gamma: float,
        lambd: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Function that computes advantages and returns from rewards and values.
        Calculated as in the original PPO paper: https://arxiv.org/abs/1707.06347
        Note that rewards may include a KL divergence loss term.

        Advantages looks like this:
        Adv1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
              - V1 + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Returns looks like this:
        Ret1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
                   + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Input:
        - values: Tensor of shape (batch_size, response_size)
        - rewards: Tensor of shape (batch_size, response_size)

        Output:
        - advantages: Tensor of shape (batch_size, response_size)
        - returns: Tensor of shape (batch_size, response_size)
        """
        if isinstance(values, list):
            # packing samples
            # TODO: this is slow...
            advantages = []
            returns = []
            for v, r in zip(values, rewards):
                adv, ret = self.get_advantages_and_returns(v.unsqueeze(0), r.unsqueeze(0), action_mask, gamma, lambd)
                advantages.append(adv.squeeze(0))
                returns.append(ret.squeeze(0))
            return advantages, returns

        lastgaelam = 0
        advantages_reversed = []
        response_length = rewards.size(1)

        # Mask invalid responses
        if action_mask is not None:
            values = action_mask * values
            rewards = action_mask * rewards

        for t in reversed(range(response_length)):
            nextvalues = values[:, t + 1] if t < response_length - 1 else 0.0
            delta = rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lambd * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values
        return advantages.detach(), returns

    @torch.no_grad()
    def get_cumulative_returns(
        self,
        rewards: torch.Tensor,
        action_mask: torch.Tensor,
        gamma: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Function that computes advantages and returns from rewards using REINFORCE.
        REINFORCE uses cumulative returns without the GAE (Generalized Advantage Estimation).

        Input:
        - rewards: Tensor of shape (batch_size, response_size)
        - action_mask: Tensor of shape (batch_size, response_size), binary mask
        - gamma: discount factor

        Output:
        - returns: Tensor of shape (batch_size, response_size)
        """

        if isinstance(rewards, list):
            # packing samples
            # TODO: this is slow...
            returns = []
            for r in rewards:
                ret = self.get_cumulative_returns(r.unsqueeze(0), action_mask, gamma)
                returns.append(ret.squeeze(0))
            return returns

        response_length = rewards.size(1)
        returns = torch.zeros_like(rewards)
        cumulative_return = torch.zeros(rewards.size(0), device=rewards.device)

        # Mask invalid responses if action_mask is provided
        if action_mask is not None:
            rewards = action_mask * rewards

        # Calculate returns by accumulating discounted rewards
        for t in reversed(range(response_length)):
            cumulative_return = rewards[:, t] + gamma * cumulative_return
            returns[:, t] = cumulative_return

        return returns  


class NaiveExperienceMakerPRM800K_BOX(ABC):
    """
    Naive experience maker.
    """

    def __init__(
        self,
        actor: Actor,
        critic: nn.Module,
        reward_model: nn.Module,
        initial_model: Actor,
        tokenizer,
        prompt_max_len: int,
        kl_controller,
        strategy=None,
        remote_rm_url: list[str] = None,
        reward_fn=None,
    ) -> None:
        super().__init__()
        self.actor = actor
        self.critic = critic
        self.reward_model = reward_model
        self.remote_rm_url = remote_rm_url
        self.initial_model = initial_model
        self.tokenizer = tokenizer
        self.prompt_max_len = prompt_max_len
        self.kl_ctl = kl_controller
        self.strategy = strategy
        self.reward_fn = reward_fn
        self.perf_stats = None
        self.advantage_estimator = strategy.args.advantage_estimator

        # custom reward func for reinforced finetuning
        self.custom_reward_func = None
        if remote_rm_url and remote_rm_url[0].endswith(".py"):
            print(f"Loading custom `reward_func(queries, prompts, labels)` from {remote_rm_url[0]}")
            import importlib.util

            spec = importlib.util.spec_from_file_location("reward_func", remote_rm_url[0])
            reward_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(reward_module)
            self.custom_reward_func = reward_module.reward_func

    # tokenizer
    def tokenize_fn(self, texts, max_length, padding=True, device=None):
        if not padding:
            # when padding is False, return tokenized texts as list
            return self.tokenizer(
                texts,
                add_special_tokens=False,
                max_length=max_length,
                truncation=True,
            )
        batch = self.tokenizer(
            texts,
            return_tensors="pt",
            add_special_tokens=False,
            max_length=max_length,
            padding=True,
            truncation=True,
        )
        return {k: v.to(device) for k, v in batch.items()}

    @torch.no_grad()
    def make_experience_list(self, all_prompts: Union[str, List[str]], all_answers:  Union[str, List[str]], all_tasks, **generate_kwargs) -> List[Experience]:
        """
        Make a list of experience with the micro_rollout_batch_size.

        This method will first calculate the response sequences and rewards for the given prompts.
        Then, if we need certain processing for the rewards or do certain filtering, we can process the rollout as a whole.
        After that, we will calculate the advantages and returns for each experience.
        """

        #print("generate_kwargs", generate_kwargs)
        args = self.strategy.args
        experiences = []
        reward_indexs_list = []
        step_rewards_list = []
        
        #print("all_prompt", all_prompts[0])
        # for samples in tqdm(
        #     self.generate_samples(all_prompts, **generate_kwargs),
        #     desc="make_experience",
        #     disable=not self.strategy.is_rank_0(),
        # ):
        #all_answers = sum([[answer] * args.n_samples_per_prompt for answer in all_answers], [])
        for idx, samples in enumerate(tqdm(
            self.generate_samples(all_prompts, all_answers, all_tasks,  **generate_kwargs),
            desc=f"make_experience",
            disable=not self.strategy.is_rank_0(),
            )):
            # 获取对应的 answer
            #answers = all_answers[idx]
            #print("samples", samples.)
            experience, reward_indexs, step_rewards = self.make_experience(samples, **generate_kwargs)
            experiences.append(experience)
            reward_indexs_list.append(reward_indexs)
            step_rewards_list.append(step_rewards)
            
            # print("experiences", experiences)
            # print("reward_indexs_list",reward_indexs_list)
            # print("step_rewards_list", step_rewards_list)

        experiences, rewards = self.process_experiences(experiences)

        # calculate return and advantages
        for e_idx, experience in enumerate(experiences):
            num_actions = experience.info["num_actions"]
            #print("middle_action_mask", experience.action_mask.shape)
            #print("middle_reward_index", reward_indexs)
            reward = compute_step_reward(
                # experience.info["reward"],
                rewards[e_idx],
                self.kl_ctl.value,
                experience.kl,
                action_mask=experience.action_mask,
                num_actions=num_actions,
                reward_clip_range=args.reward_clip_range,
                reward_indexs=reward_indexs_list[e_idx],
                step_rewards=step_rewards_list[e_idx]
                
            )

            if self.advantage_estimator == "gae":
                experience.advantages, experience.returns = self.get_advantages_and_returns(
                    experience.values,
                    reward,
                    experience.action_mask,
                    generate_kwargs["gamma"],
                    generate_kwargs["lambd"],
                )
            elif self.advantage_estimator in ["reinforce", "rloo", "reinforce_baseline", "group_norm"]:
                experience.returns = self.get_cumulative_returns(
                    reward,
                    experience.action_mask,
                    generate_kwargs["gamma"],
                )
                experience.advantages = deepcopy(experience.returns)
            else:
                raise Exception(f"Unkown advantage_estimator {self.advantage_estimator}")

            # calculate the return info.
            if not getattr(self, "packing_samples", False):
                return_sums = reward.sum(dim=-1)
            else:
                return_sums = torch.tensor(
                    [each_reward.sum() for each_reward in reward], device=torch.cuda.current_device()
                )
            experience.info["return"] = return_sums
            # remove unnecessary info
            experience.kl = None
            del experience.info["num_actions"]
            experience.to_device("cpu")
        return experiences

    @torch.no_grad()
    def generate_samples(self, all_prompts: List[str], all_answers: List[str], all_tasks, **generate_kwargs) -> List[Samples]:
        """
        Generate samples and return in batches.
        """
        assert not getattr(self, "packing_samples", False)
        args = self.strategy.args
        self.actor.eval()
        # sample multiple response
        all_prompts = sum([[prompt] * args.n_samples_per_prompt for prompt in all_prompts], [])
        all_answers = sum([[answer] * args.n_samples_per_prompt for answer in all_answers], [])
        all_tasks = sum([[task] * args.n_samples_per_prompt for task in all_tasks], [])
        samples_list = []
        #self.strategy.print(f"generate samples!!!")
        for i in range(0, len(all_prompts), args.micro_rollout_batch_size):
            prompts = all_prompts[i : i + args.micro_rollout_batch_size]
            answers = all_answers[i : i + args.micro_rollout_batch_size]
            tasks = all_tasks[i : i + args.micro_rollout_batch_size]
            inputs = self.tokenize_fn(prompts, self.prompt_max_len, device="cuda")
            #self.strategy.print(f"generating!!!")
            sequences, attention_mask, action_mask = self.actor.generate(**inputs, **generate_kwargs)
            #self.strategy.print(f"generating!!!", sequences)
            samples = SamplesBOX(
                sequences=sequences,
                attention_mask=attention_mask,
                action_mask=action_mask,
                num_actions=action_mask.size(1),
                packed_seq_lens=None,
                response_length=action_mask.float().sum(dim=-1),
                total_length=attention_mask.float().sum(dim=-1),
                prompts=prompts,
                answers=answers,
                tasks=tasks,
            )
            samples_list.append(samples)
            #print("sequences", samples.sequences)
            #print("attention_mask", samples.attention_mask)

        return samples_list

    @torch.no_grad()
    def make_experience(self, samples: Samples, **generate_kwargs) -> Experience:
        """
        Turn samples into experience by calculating logprobs, values, rewards, and kl divergence.
        """
        self.actor.eval()
        if self.initial_model is not None:
            self.initial_model.eval()
        if self.reward_model is not None:
            self.reward_model.eval()
        if self.critic is not None:
            self.critic.eval()

        # extract values from samples
        sequences = samples.sequences
        attention_mask = samples.attention_mask
        action_mask = samples.action_mask
        num_actions = samples.num_actions
        answers = samples.answers
        tasks = samples.tasks

        # log probs
        #print("sequences.shape", sequences.shape)
        #print("sequences", sequences[0])
        #print("num_actions", num_actions)
        #print("attention_mask", attention_mask.shape)
        #print("action_mask", action_mask.shape)
        action_log_probs = self.actor(sequences, num_actions, attention_mask)
        
        #print("action_log_probs", action_log_probs.shape)

        # init log probs
        # init log probs
        if self.initial_model is not None:
            base_action_log_probs = self.initial_model(sequences, num_actions, attention_mask)
        else:
            base_action_log_probs = None

        # values
        if self.critic is not None:
            value = self.critic(sequences, num_actions, attention_mask)
        else:
            value = None

        # rewards
        if self.remote_rm_url is not None:
            # remote RM
            queries = self.tokenizer.batch_decode(sequences.cpu(), skip_special_tokens=False)
            if self.custom_reward_func:
                r = self.custom_reward_func(queries, samples.prompts, samples.labels).to(
                    device=action_log_probs.device
                )
        else:
            # local RM
            
            #print("sequences", sequences[0])

            queries = self.tokenizer.batch_decode(sequences.cpu(), skip_special_tokens=False)
            
            
            #print("queries", queries[0])

            # 监测不包含 \n\n# Answer\n\n 的 queries
            
            #print("queries", queries[0])
            # 定义正则模式
            #pattern = r"The final answer is: \\\\boxed\{(.*?)\}"
            pattern = r"The final answer is: \\boxed\{(.*?)\}"
            # 找到没有匹配到内容的索引
            missing_answer_indices = [
            i for i, query in enumerate(queries) if not re.search(pattern, query, re.DOTALL)
            ]
            

            processed_queries = []
            box_match_list = []
            for query, answer in zip(queries, answers):
                query, box_match = preprocess_orm800k_box_responsev1(query, answer)
                processed_queries.append(query)
                box_match_list.append(box_match)
                
            queries = processed_queries
            
            
            reward_sequences, reward_attention_mask = preprocess_prm_reward(queries, self.tokenizer, self.prompt_max_len, **generate_kwargs)
            

            candidate_tokens = [128003, 128004]
            logits = self.reward_model(reward_sequences.to(device=action_log_probs.device), attention_mask=reward_attention_mask.to(device=action_log_probs.device), return_output=True).logits[:, :, candidate_tokens]
            scores = logits.softmax(dim=-1)[:, :, 0]
            step_scores = []
            step_tag = '<|reserved_special_token_0|>'
            step_tag_id = self.tokenizer.encode(f"{step_tag}")[-1]

            for index, score in enumerate(scores):
                if index in missing_answer_indices:
                # 如果 query 不包含 \n\n# Answer\n\n，设置 step_score 为 0.0
                    step_scores.append(torch.tensor(0.0, device=action_log_probs.device))
                else:
                    # 否则正常计算 step_score
                    step_scores.append(score[reward_sequences[index] == step_tag_id])
                    
            reward_indexs = []
            
            for seq in sequences:
                indices = (seq == step_tag_id).nonzero(as_tuple=True)[0]
                if len(indices) > 0:
                    #print("neg num actions", -num_actions)
                    reverse_indices = [- (len(seq) - idx.item()) for idx in indices]
                    #print("reverse_indices", reverse_indices)
                    
                else:
                    reverse_indices = [-1]
                    
                reward_indexs.append(torch.tensor(reverse_indices, device=action_log_probs.device))
                
                
            # 使用负索引提取128002的位置
            extracted_elements = []
            for seq, negative_indices in zip(sequences, reward_indexs):
                elements = [seq[idx].item() for idx in negative_indices]
                extracted_elements.append(elements)

            # 提取每个张量的最小值
            #min_values = [step_score.min() for step_score in step_scores]

            # 提取每个张量的最小值，若为空则设置为0.0
            min_values = [step_score.min()-0.5 if step_score.numel() > 0 else torch.tensor(-0.5, device=action_log_probs.device) for step_score in step_scores]
            # 将最小值重新合并为一个新的张量
            min_values_tensor = torch.tensor(min_values, device=action_log_probs.device)

            r = min_values_tensor.to(device=action_log_probs.device)
            
            step_scores = [step_score - 0.5 if step_score.numel() > 0 else torch.tensor(-0.5, device=action_log_probs.device) for step_score in step_scores]
            
            # 将 math_box_list 转换为张量，并移动到与 step_scores 相同的设备
            math_box_tensor = torch.tensor(box_match_list, device=r.device)
            # print("r", r)
            
            # print("step_scores", step_scores)
            if len(math_box_tensor) == len(step_scores):
                
                r = r + math_box_tensor
                for i in range(len(math_box_tensor)):
                    step_score = step_scores[i]
                    add_box = math_box_tensor[i]
                    
                    if step_score.dim() == 0:
                        step_scores[i] = step_score + add_box
                    else:
                        step_scores[i][-1] += add_box
                        
        if (self.initial_model is not None) and (not self.strategy.args.use_kl_loss):   
            kl = compute_approx_kl(
                action_log_probs,
                base_action_log_probs,
                action_mask=action_mask,
                use_kl_estimator_k3=self.strategy.args.use_kl_estimator_k3,
            )
            #print("kl.shape", kl.shape)
        else:
            kl = torch.zeros_like(action_log_probs, dtype=action_log_probs.dtype, device=action_log_probs.device)
        
        for i, tensor in enumerate(reward_indexs):
            if not torch.equal(tensor, torch.tensor([-1], device=action_log_probs.device)):
                # Remove the second last element
                reward_indexs[i] = torch.cat((tensor[:-2], tensor[-1:]))
            
        # print(masked_mean(kl, action_mask, dim=-1).shape)
        info = {
            "kl": masked_mean(kl, action_mask, dim=-1),
            "reward": r,
            "response_length": samples.response_length,
            "total_length": samples.total_length,
            "num_actions": num_actions,
            #"reward_indexs": reward_indexs,
            #"step_rewards": step_scores,
        }
        # reset model state
        self.actor.train()
        if self.critic is not None:
            self.critic.train()
            
        # print("pre_pre_action_mask", action_mask.size(1))

        return Experience(
            sequences,
            action_log_probs,
            base_action_log_probs,
            value,
            None,
            None,
            attention_mask,
            action_mask,
            info,
            kl,
        ), reward_indexs, step_scores

    @torch.no_grad()
    def process_experiences(self, experiences: List[Experience]) -> Tuple[List[Experience], List[torch.Tensor]]:
        """
        Process experiences, this can be used to filter out some experiences or do some processing on the rewards.

        Output:
        - experiences: List of Experience
        - rewards: List of rewards
        """
        # return experiences
        args = self.strategy.args
        # reward shaping for rloo and reinforce_baseline
        if args.advantage_estimator == "rloo":
            rewards = torch.cat([experience.info["reward"] for experience in experiences])
            rewards = rewards.reshape(-1, args.n_samples_per_prompt).to(device="cuda")
            baseline = (rewards.sum(-1, keepdim=True) - rewards) / (args.n_samples_per_prompt - 1)
            rewards = rewards - baseline
            rewards = rewards.flatten().to(device="cpu").chunk(len(experiences))
            return experiences, rewards
        elif args.advantage_estimator == "reinforce_baseline":
            # REINFORCE++-baseline removed the / std and K3 kl loss in GRPO.
            # `/ std` is not needed in RL variance reduction theory, and `k3 KL` has a larger variance than `k1 KL` under a categorical distribution.
            rewards = torch.cat([experience.info["reward"] for experience in experiences])
            rewards = rewards.reshape(-1, args.n_samples_per_prompt).to(device="cuda")
            rewards = rewards - rewards.mean(-1, keepdim=True)
            rewards = rewards.reshape(-1).to(device="cpu").chunk(len(experiences))
            return experiences, rewards
        elif args.advantage_estimator == "group_norm":
            rewards = torch.cat([experience.info["reward"] for experience in experiences])
            rewards = rewards.reshape(-1, args.n_samples_per_prompt).to(device="cuda")
            rewards = (rewards - rewards.mean(-1, keepdim=True)) / (rewards.std(-1, keepdim=True) + 1e-9)
            rewards = rewards.reshape(-1).to(device="cpu").chunk(len(experiences))
            return experiences, rewards
        # default rewards
        return experiences, [experience.info["reward"] for experience in experiences]
    

    @torch.no_grad()
    def get_advantages_and_returns(
        self,
        values: torch.Tensor,
        rewards: torch.Tensor,
        action_mask: torch.Tensor,
        gamma: float,
        lambd: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Function that computes advantages and returns from rewards and values.
        Calculated as in the original PPO paper: https://arxiv.org/abs/1707.06347
        Note that rewards may include a KL divergence loss term.

        Advantages looks like this:
        Adv1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
              - V1 + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Returns looks like this:
        Ret1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
                   + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Input:
        - values: Tensor of shape (batch_size, response_size)
        - rewards: Tensor of shape (batch_size, response_size)

        Output:
        - advantages: Tensor of shape (batch_size, response_size)
        - returns: Tensor of shape (batch_size, response_size)
        """
        if isinstance(values, list):
            # packing samples
            # TODO: this is slow...
            advantages = []
            returns = []
            for v, r in zip(values, rewards):
                adv, ret = self.get_advantages_and_returns(v.unsqueeze(0), r.unsqueeze(0), action_mask, gamma, lambd)
                advantages.append(adv.squeeze(0))
                returns.append(ret.squeeze(0))
            return advantages, returns

        lastgaelam = 0
        advantages_reversed = []
        response_length = rewards.size(1)

        # Mask invalid responses
        if action_mask is not None:
            values = action_mask * values
            rewards = action_mask * rewards

        for t in reversed(range(response_length)):
            nextvalues = values[:, t + 1] if t < response_length - 1 else 0.0
            delta = rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lambd * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values
        return advantages.detach(), returns

    @torch.no_grad()
    def get_cumulative_returns(
        self,
        rewards: torch.Tensor,
        action_mask: torch.Tensor,
        gamma: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Function that computes advantages and returns from rewards using REINFORCE.
        REINFORCE uses cumulative returns without the GAE (Generalized Advantage Estimation).

        Input:
        - rewards: Tensor of shape (batch_size, response_size)
        - action_mask: Tensor of shape (batch_size, response_size), binary mask
        - gamma: discount factor

        Output:
        - returns: Tensor of shape (batch_size, response_size)
        """

        if isinstance(rewards, list):
            # packing samples
            # TODO: this is slow...
            returns = []
            for r in rewards:
                ret = self.get_cumulative_returns(r.unsqueeze(0), action_mask, gamma)
                returns.append(ret.squeeze(0))
            return returns

        response_length = rewards.size(1)
        returns = torch.zeros_like(rewards)
        cumulative_return = torch.zeros(rewards.size(0), device=rewards.device)

        # Mask invalid responses if action_mask is provided
        if action_mask is not None:
            rewards = action_mask * rewards

        # Calculate returns by accumulating discounted rewards
        for t in reversed(range(response_length)):
            cumulative_return = rewards[:, t] + gamma * cumulative_return
            returns[:, t] = cumulative_return

        return returns


class RemoteExperienceMaker(NaiveExperienceMaker):
    def __init__(self, *args, vllm_engines: List = None, packing_samples=False, vllm_engines_fixed: List = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.vllm_engines = vllm_engines
        self.packing_samples = packing_samples
        self.vllm_engines_fixed = vllm_engines_fixed
        if self.custom_reward_func:
            self.custom_reward_func = ray.remote(self.custom_reward_func)

    @torch.no_grad()
    def make_experience_list(
        self, all_prompts: Union[str, List[str]], all_labels, all_tasks, **generate_kwargs
    ) -> List[Experience]:
        if self.strategy.args.perf:
            self.perf_stats = {
                "generate_time": 0,
                "actor_value_rm_time": 0,
                "wait_time": 0,
            }
        experiences = super().make_experience_list(all_prompts, all_labels, all_tasks, **generate_kwargs)
        if self.critic is not None:
            for experience in experiences:
                # send experience to critic
                experience_cpu = deepcopy(experience)
                experience_cpu.to_device("cpu")
                self._ref = self.critic.append.remote(experience_cpu)
        return experiences

    @torch.no_grad()
    def generate_samples(self, all_prompts: List[str], all_labels, all_tasks, **generate_kwargs) -> List[Samples]:
        """
        Generate samples and return in batches.

        When not using vllm, we will fallback to the default implementation,
        in which actor will be used to generate samples.
        """
        if self.vllm_engines is None:
            return super().generate_samples(all_prompts, all_labels, all_tasks, **generate_kwargs)

        # vLLM generation
        samples = self._generate_vllm(all_prompts, all_labels, all_tasks, **generate_kwargs)

        # vLLM offload when colocate_all_models
        if self.strategy.args.vllm_enable_sleep:
            if torch.distributed.get_rank() == 0:
                refs = []
                for engine in self.vllm_engines:
                    refs.append(engine.sleep.remote())
                ray.get(refs)
        return samples

    @torch.no_grad()
    def make_experience(self, samples: Samples) -> Experience:
        """
        Turn samples into experience by calculating logprobs, values, rewards, and kl divergence.
        """
        args = self.strategy.args
        self.actor.eval()
        device = torch.cuda.current_device()
        # extract values from samples
        sequences = samples.sequences
        attention_mask = samples.attention_mask
        action_mask = samples.action_mask
        num_actions = samples.num_actions
        packed_seq_lens = samples.packed_seq_lens
        tasks = samples.tasks
        start = time.time()
        sequences_cpu, attention_mask_cpu = (
            sequences.to("cpu"),
            attention_mask.to("cpu"),
        )

        # init log probs
        if self.initial_model is not None:
            base_action_log_probs_ref = self.initial_model.forward.remote(
                sequences_cpu, num_actions, attention_mask_cpu, packed_seq_lens=packed_seq_lens
            )

            if args.colocate_actor_ref or args.colocate_all_models:
                ray.get([base_action_log_probs_ref])
                ray.get([self.initial_model.empty_cache.remote()])
        else:
            base_action_log_probs_ref = ray.put(None)

        # values
        if self.critic is not None:
            value_ref = self.critic.forward.remote(
                sequences_cpu, num_actions, attention_mask_cpu, packed_seq_lens=packed_seq_lens
            )
            # avoid CUDA OOM when colocate models
            if args.colocate_critic_reward or args.colocate_all_models:
                ray.get([value_ref])
                ray.get([self.critic.empty_cache.remote()])
        else:
            value_ref = ray.put(None)

        # rewards
        r_refs = []
        # support remote RM API with ray
        if not self.remote_rm_url:
            for rm in self.reward_model:
                r_refs.append(rm.forward.remote(sequences_cpu, attention_mask_cpu, packed_seq_lens=packed_seq_lens))
        else:
            # remote RM
            if not self.packing_samples:
                queries = self.tokenizer.batch_decode(sequences_cpu, skip_special_tokens=False)
            else:
                sequences_list = []
                offset = 0
                tokens_list = sequences_cpu.tolist()[0]
                for length in packed_seq_lens:
                    sequences_list.append(tokens_list[offset : offset + length])
                    offset += length
                queries = self.tokenizer.batch_decode(sequences_list, skip_special_tokens=False)

            if self.custom_reward_func:
                r = self.custom_reward_func.remote(queries, samples.prompts, samples.labels)
                r_refs.append(r)
            else:
                for rm in self.remote_rm_url:
                    r = remote_rm_fn_ray.remote(rm, queries=queries, prompts=samples.prompts, labels=samples.labels)
                    r_refs.append(r)

        if args.colocate_all_models and not self.remote_rm_url:
            ray.get(r_refs)
            ray.get([self.reward_model[0].empty_cache.remote()])

        # log probs
        action_log_probs = self.actor(sequences, num_actions, attention_mask, packed_seq_lens=packed_seq_lens)
        actor_value_rm_time = time.time() - start

        # wait initial/critic/reward model done
        start = time.time()
        ref_values = ray.get([base_action_log_probs_ref, value_ref] + r_refs)
        wait_time = time.time() - start

        base_action_log_probs, value, rewards = ref_values[0], ref_values[1], ref_values[2:]
        if base_action_log_probs is not None:
            base_action_log_probs = base_action_log_probs.to(device)
        if value is not None:
            value = value.to(device)
        rewards = [r.to(device) for r in rewards]
        r = self.reward_fn(rewards) if len(rewards) > 0 else rewards[0]

        # avoid CUDA OOM when colocate models
        if args.colocate_critic_reward and not self.remote_rm_url:
            ray.get([self.reward_model[0].empty_cache.remote()])

        if args.colocate_actor_ref or args.colocate_all_models:
            torch.cuda.empty_cache()

        if (self.initial_model is not None) and (not args.use_kl_loss):
            kl = compute_approx_kl(
                action_log_probs,
                base_action_log_probs,
                action_mask=action_mask,
                use_kl_estimator_k3=args.use_kl_estimator_k3,
            )
        else:
            kl = torch.zeros_like(action_log_probs, dtype=action_log_probs.dtype, device=device)

        if not self.packing_samples:
            kl_mean = masked_mean(kl, action_mask, dim=-1)
        else:
            # convert tensor into list of tensors so that it's easier to manipulate
            # within dataset.
            sequences = unpacking_samples(sequences, packed_seq_lens)
            attention_mask = None
            action_log_probs = unpacking_samples(action_log_probs, num_actions)
            if value is not None:
                value = unpacking_samples(value, num_actions)
            if base_action_log_probs is not None:
                base_action_log_probs = unpacking_samples(base_action_log_probs, num_actions)

            kl = unpacking_samples(kl, num_actions)
            kl_mean = torch.tensor([each_kl.mean() for each_kl in kl], device=device)

        if not args.use_kl_loss:
            base_action_log_probs = None

        info = {
            "kl": kl_mean,
            "reward": r,
            "response_length": samples.response_length,
            "total_length": samples.total_length,
            "num_actions": num_actions,
        }
        if self.strategy.args.perf:
            self.perf_stats["actor_value_rm_time"] += actor_value_rm_time
            self.perf_stats["wait_time"] += wait_time

        experience = Experience(
            sequences,
            action_log_probs,
            base_action_log_probs,
            value,
            None,
            None,
            attention_mask,
            action_mask,
            info,
            kl,
        )

        self.actor.train()  # reset model state
        return experience

    def _generate_vllm(self, all_prompts: List[str], all_labels, all_tasks, **kwargs) -> List[Samples]:
        from vllm import SamplingParams

        # round-robin load balance
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()

        # Select LLM engines: assign each rank an engine, or cycle through engines if world_size < engine_count
        if len(self.vllm_engines) <= world_size:
            llms = [self.vllm_engines[rank % len(self.vllm_engines)]]
        else:
            llms = self.vllm_engines[rank::world_size]

        args = self.strategy.args

        sampling_params = SamplingParams(
            temperature=kwargs.get("temperature", 1.0),
            top_p=kwargs.get("top_p", 1.0),
            top_k=kwargs.get("top_k", -1),
            max_tokens=kwargs.get("max_new_tokens", 1024),
            min_tokens=kwargs.get("min_new_tokens", 1),
            skip_special_tokens=kwargs.get("skip_special_tokens", False),
            include_stop_str_in_output=True,
        )

        # Expand prompt list based on the number of samples per prompt
        all_prompts = sum([[prompt] * args.n_samples_per_prompt for prompt in all_prompts], [])
        all_labels = sum([[label] * args.n_samples_per_prompt for label in all_labels], [])
        all_tasks = sum([[task] * args.n_samples_per_prompt for task in all_tasks], [])
        all_prompt_token_ids = self.tokenize_fn(all_prompts, self.prompt_max_len, padding=False)["input_ids"]

        # Distribute requests to engines and collect responses to outputs
        refs = []
        batch_size = (len(all_prompt_token_ids) + len(llms) - 1) // len(llms)
        for i, llm in enumerate(llms):
            prompt_token_ids = all_prompt_token_ids[i * batch_size : (i + 1) * batch_size]
            refs.append(
                llm.add_requests.remote(rank, sampling_params=sampling_params, prompt_token_ids=prompt_token_ids)
            )
        ray.get(refs)

        # Make sure all requests are sent.
        torch.distributed.barrier()

        # Retrieve and combine results from all outputs
        all_output_refs = []
        for i, llm in enumerate(llms):
            all_output_refs.append(llm.get_responses.remote(rank))
        all_outputs = sum(ray.get(all_output_refs), [])

        samples_list = []
        for i in range(0, len(all_outputs), args.micro_rollout_batch_size):
            outputs = all_outputs[i : i + self.strategy.args.micro_rollout_batch_size]
            prompts = all_prompts[i : i + self.strategy.args.micro_rollout_batch_size]
            labels = all_labels[i : i + self.strategy.args.micro_rollout_batch_size]
            tasks = all_tasks[i : i + self.strategy.args.micro_rollout_batch_size]
            if not self.packing_samples:
                # NOTE: concat all outputs to following format:
                #
                # | [PAD] [PAD] token token token | token token [EOS] [PAD] |
                # | token token token token token | token token [EOS] [PAD] |
                # | [PAD] [PAD] [PAD] token token | token token token [EOS] |
                # |<---------- prompt ----------->|<-------- answer ------->|
                max_input_len, max_output_len = 0, 0
                for output in outputs:
                    max_input_len = max(max_input_len, len(output.prompt_token_ids))
                    max_output_len = max(max_output_len, len(output.outputs[0].token_ids))

                pad_token_id, eos_token_id = self.tokenizer.pad_token_id, self.tokenizer.eos_token_id
                sequences = []
                for output in outputs:
                    # left padding input
                    input_len = len(output.prompt_token_ids)
                    input_ids = [pad_token_id] * (max_input_len - input_len) + list(output.prompt_token_ids)

                    # right padding output
                    output_len = len(output.outputs[0].token_ids)
                    output_ids = list(output.outputs[0].token_ids) + [pad_token_id] * (max_output_len - output_len)

                    # concat input and output
                    sequences.append(input_ids + output_ids)

                sequences = torch.tensor(sequences)
                sequences, attention_mask, action_mask = self.actor.process_sequences(
                    sequences, max_input_len, eos_token_id, pad_token_id
                )
                sequences = sequences.to("cuda")
                attention_mask = attention_mask.to("cuda")
                action_mask = action_mask.to("cuda")
                samples_list.append(
                    Samples(
                        sequences=sequences,
                        attention_mask=attention_mask,
                        action_mask=action_mask,
                        num_actions=action_mask.size(1),
                        packed_seq_lens=None,
                        response_length=action_mask.float().sum(dim=-1),
                        total_length=attention_mask.float().sum(dim=-1),
                        prompts=prompts,
                        answers=labels,
                        tasks=tasks
                    )
                )
            else:
                # NOTE: concat all outputs to following format:
                #
                # | token token token | token token [EOS] | token token token token token | token token [EOS] | token token | token token token [EOS] |
                # |<---  prompt ----->|<---- answer ----->|<---------- prompt ----------->|<----- answer ---->|<- prompt -->|<-------- answer ------->|
                pad_token_id, eos_token_id = self.tokenizer.pad_token_id, self.tokenizer.eos_token_id
                sequences = []
                packed_seq_lens = []
                attention_mask = []
                num_actions = []
                for i, output in enumerate(outputs):
                    input_len = len(output.prompt_token_ids)
                    output_len = len(output.outputs[0].token_ids)
                    packed_seq_lens.append(input_len + output_len)
                    sequences.extend(output.prompt_token_ids + list(output.outputs[0].token_ids))
                    attention_mask.extend([i + 1] * (input_len + output_len))

                    # current_action_mask = [0] * (input_len - 1) + [1] * output_len + [0]
                    # num_actions.append(max(1, sum(current_action_mask)))
                    num_actions.append(max(1, output_len))

                sequences = torch.tensor(sequences, device="cuda").unsqueeze(0)
                attention_mask = torch.tensor(attention_mask, device="cuda").unsqueeze(0)
                action_mask = None
                response_length = torch.tensor(num_actions, device="cuda", dtype=torch.float)
                total_length = torch.tensor(packed_seq_lens, device="cuda", dtype=torch.float)
                samples_list.append(
                    Samples(
                        sequences=sequences,
                        attention_mask=attention_mask,
                        action_mask=None,
                        num_actions=num_actions,
                        packed_seq_lens=packed_seq_lens,
                        response_length=response_length,
                        total_length=total_length,
                        prompts=prompts,
                        answers=labels,
                        tasks=tasks,
                    )
                )
        return samples_list

    def flush(self):
        "Ensure all experience has been send to critic"
        if self.critic is not None:
            ray.get(self._ref)
            self._ref = None


            
class RemoteExperienceMakerBOX(NaiveExperienceMakerBOX):
    def __init__(self, *args, vllm_engines: List = None, packing_samples=False, vllm_engines_fixed: List = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.vllm_engines = vllm_engines
        self.packing_samples = packing_samples
        self.vllm_engines_fixed = vllm_engines_fixed
        if self.strategy.args.reward_pretrain:
            self.tokenizer_reward = AutoTokenizer.from_pretrained(self.strategy.args.reward_pretrain)
        else:
            self.tokenizer_reward = None

        if self.custom_reward_func:
            self.custom_reward_func = ray.remote(self.custom_reward_func)

    @torch.no_grad()
    def make_experience_list(
        self, all_prompts: Union[str, List[str]], all_labels, all_tasks, **generate_kwargs
    ) -> List[Experience]:
        if self.strategy.args.perf:
            self.perf_stats = {
                "generate_time": 0,
                "actor_value_rm_time": 0,
                "wait_time": 0,
            }
        experiences = super().make_experience_list(all_prompts, all_labels, all_tasks, **generate_kwargs)
        if self.critic is not None:
            for experience in experiences:
                # send experience to critic
                experience_cpu = deepcopy(experience)
                experience_cpu.to_device("cpu")
                self._ref = self.critic.append.remote(experience_cpu)
        return experiences

    @torch.no_grad()
    def generate_samples(self, all_prompts: List[str], all_labels, all_tasks, **generate_kwargs) -> List[Samples]:
        """
        Generate samples and return in batches.

        When not using vllm, we will fallback to the default implementation,
        in which actor will be used to generate samples.
        """
        if self.vllm_engines is None:
            return super().generate_samples(all_prompts, all_answers, all_tasks, **generate_kwargs)

        # vLLM generation
        samples = self._generate_vllm(all_prompts, all_labels, all_tasks, **generate_kwargs)

        # vLLM offload when colocate_all_models
        if self.strategy.args.vllm_enable_sleep:
            if torch.distributed.get_rank() == 0:
                refs = []
                for engine in self.vllm_engines:
                    refs.append(engine.sleep.remote())
                ray.get(refs)
        return samples

    @torch.no_grad()
    def make_experience(self, samples: Samples) -> Experience:
        """
        Turn samples into experience by calculating logprobs, values, rewards, and kl divergence.
        """
        args = self.strategy.args
        self.actor.eval()
        device = torch.cuda.current_device()
        # extract values from samples
        sequences = samples.sequences
        attention_mask = samples.attention_mask
        action_mask = samples.action_mask
        num_actions = samples.num_actions
        packed_seq_lens = samples.packed_seq_lens
        answers = samples.answers
        tasks = samples.tasks
        generate_kwargs = samples.generate_kwargs
        start = time.time()
        prompts = samples.prompts
        sequences_cpu, attention_mask_cpu = (
            sequences.to("cpu"),
            attention_mask.to("cpu"),
        )
        # ray.logger.info(f"sequences_len: {sequences_cpu.shape}")
        # init log probs
        if self.initial_model is not None:
            base_action_log_probs_ref = self.initial_model.forward.remote(
                sequences_cpu, num_actions, attention_mask_cpu, packed_seq_lens=packed_seq_lens
            )
            if args.colocate_actor_ref or args.colocate_all_models:
                ray.get([base_action_log_probs_ref])
                ray.get([self.initial_model.empty_cache.remote()])
        else:
            base_action_log_probs_ref = ray.put(None)

        # values
        # ray.logger.info("sent initial model forward request")
        if self.critic is not None:
            value_ref = self.critic.forward.remote(
                sequences_cpu, num_actions, attention_mask_cpu, packed_seq_lens=packed_seq_lens
            )
            # avoid CUDA OOM when colocate models
            if args.colocate_critic_reward or args.colocate_all_models:
                ray.get([value_ref])
                ray.get([self.critic.empty_cache.remote()])
        else:
            value_ref = ray.put(None)
        
        # ray.logger.info("sent value model forward request")
        if self.strategy.args.colocate_actor_ref:
            ray.get([base_action_log_probs_ref])
            ray.get([self.initial_model.empty_cache.remote()])

        # rewards
        r_refs = []
        r_refs_format = []
        r_refs_answer = []
        all_format_correct_list = []
        all_answer_correct_list = []
        model_output_experience_list = []
        all_loss_mask_enforce_superior_list = []
        # print("answers", len(answers))
        # print("sequences_cpu", sequences_cpu.shape)
        # support remote RM API with ray
        if not self.remote_rm_url:
            queries = self.tokenizer.batch_decode(sequences.cpu(), skip_special_tokens=False)
            processed_queries = []
            box_match_list = []
            format_match_list = []
            all_match_list = []
            format_correct_list = []
            answer_correct_list = []
            complex_scoring_candidates = []
            repetition_penalty_list = []
            loss_mask_enforce_superior_list = []

            ray.logger.info("Processing rule-based rewards and complex scoring candidates")
            rule_based_start = time.time()
            # First process all samples with basic rule-based rewards
            def process_rule_based_rewards(query_idx, query, answer, task):
                if task.startswith("complex"):
                    query, box_match, is_answer_correct, num_actions_judgement, wrapping_content_list_all, format_match, is_format_correct, wrapping_pairs_list_all, model_input, \
                        model_output, scoring_questions, repetition_penalty, reference_scoring_woCoT = preprocess_box_response_for_qwen_prompt(deepcopy(query), deepcopy(answer), deepcopy(task),\
                            self.tokenizer, self.tokenizer_reward, args.generate_max_len, args, generate_kwargs, packed_seq_lens)
                    
                    ray.logger.info(f"[TASK={task}]: answer reward={box_match}, is_answer_correct={is_answer_correct}, format_match={format_match}, is_format_correct={is_format_correct}")
                    # Queue for complex model scoring later
                    if box_match == -100 and is_answer_correct == -100 and num_actions_judgement and len(wrapping_content_list_all) and len(wrapping_pairs_list_all):
                        # repetition penalty to be added later
                        return (query, query_idx, num_actions_judgement, wrapping_content_list_all, wrapping_pairs_list_all,\
                            scoring_questions, format_match, is_format_correct, model_input, model_output, repetition_penalty, answer, task, reference_scoring_woCoT), True
                    else:
                        # simply ignore it
                        is_answer_superior_to_answer_woCoT = True
                    
                else:
                    query, box_match, format_match, is_format_correct, is_answer_correct, model_input,\
                        model_output, repetition_penalty, is_answer_superior_to_answer_woCoT = preprocess_box_response_for_qwen_prompt(deepcopy(query), deepcopy(answer), deepcopy(task),\
                        self.tokenizer, self.tokenizer_reward, args.generate_max_len, args, generate_kwargs, packed_seq_lens)

                    ray.logger.info(f"[TASK={task}]: answer reward={box_match}, is_answer_correct={is_answer_correct}, format_match={format_match}, is_format_correct={is_format_correct}")
                return (query, box_match, format_match, is_format_correct, is_answer_correct, model_input, model_output, repetition_penalty, answer, task, is_answer_superior_to_answer_woCoT), False

            rule_based_refs = []
            for query_idx, (query, answer, task) in enumerate(zip(queries, answers, tasks)):
                rule_based_refs.append(process_rule_based_rewards(deepcopy(query_idx), deepcopy(query), deepcopy(answer), deepcopy(task)))

            # Process results and collect complex scoring candidates
            for rule_based_ref in rule_based_refs:
                result, needs_complex_scoring = rule_based_ref
                if needs_complex_scoring:
                    query, query_idx, num_actions_judgement, wrapping_content_list_all, wrapping_pairs_list_all,\
                        scoring_questions, format_match, is_format_correct, model_input, model_output, repetition_penalty, answer, task, reference_scoring_woCoT = result
                        
                    complex_scoring_candidates.append((query_idx, num_actions_judgement, wrapping_content_list_all, wrapping_pairs_list_all, scoring_questions))
                    is_answer_correct = -100.0
                    box_match = -100.0
                    loss_mask_enforce_superior_list.append(reference_scoring_woCoT)

                else:
                    query, box_match, format_match, is_format_correct, is_answer_correct, model_input, model_output, repetition_penalty, answer, task, is_answer_superior_to_answer_woCoT = result
                    loss_mask_enforce_superior_list.append(is_answer_superior_to_answer_woCoT)

                processed_queries.append(query)
                box_match_list.append(box_match)
                format_match_list.append(format_match) 
                all_match_list.append(box_match + format_match)
                format_correct_list.append(is_format_correct)
                answer_correct_list.append(is_answer_correct)
                repetition_penalty_list.append(repetition_penalty)
                model_output_experience_list.append([deepcopy(model_input), deepcopy(model_output), deepcopy(answer), deepcopy(task)])
            
            rule_based_time = time.time() - rule_based_start
            ray.logger.info(f"Rule based rewards took {rule_based_time:.2f}s")

            queries = processed_queries
            if self.reward_model is not None:
                if len(complex_scoring_candidates):
                    # 需要批量推理scoring questions
                    ray.logger.info("Processing complex scoring...")
                    complex_start = time.time()
                    judge_base_action_log_probs_ref_list_all = []
                    
                    prompts_scoring_inputs, indices, wrapping_content_list_all_list, wrapping_pairs_list_all_list,\
                        num_actions_judgement = compute_judgement_score_preprocess(complex_scoring_candidates, self.tokenizer, args.generate_max_len)
                    # wrapping_content_list_all_list_str = ""
                    # for wrapping_content_list_all_list_item_idx, wrapping_content_list_all_list_item in enumerate(wrapping_content_list_all_list):
                    #     wrapping_content_list_all_list_str += f"JudgeQuestion {wrapping_content_list_all_list_item_idx}.\n\n" + str(wrapping_content_list_all_list_item)
                    #     wrapping_content_list_all_list_str += "\n\n"
                    # ray.logger.info(f"------>prompts_scoring_inputs for judgement\n\n{wrapping_content_list_all_list_str}\n\n<------")
                    judge_base_action_log_probs_ref_all = self.reward_model[0].forward.remote((prompts_scoring_inputs["input_ids"]).to("cpu"),\
                                num_actions=num_actions_judgement, attention_mask=(prompts_scoring_inputs["attention_mask"]).to("cpu"),\
                                    packed_seq_lens=packed_seq_lens, return_output=False)
                    judge_base_action_log_probs_ref_list_all = ray.get(judge_base_action_log_probs_ref_all)
                    judgement_scores = compute_judgement_score_postprocess(prompts_scoring_inputs, indices,\
                        wrapping_content_list_all_list, wrapping_pairs_list_all_list,\
                            judge_base_action_log_probs_ref_list_all, answer_reward=2.0)
                    # judgement_scores = compute_judgement_score(complex_scoring_candidates, self.initial_model,\
                    #     self.tokenizer, args.generate_max_len, args, packed_seq_lens,\
                    #         ignore_format=False, actor=None, format_reward=1.0, answer_reward=2.0)
                    
                    for judgement_score in judgement_scores:
                        query_idx, box_match, is_answer_correct, scoring_wCoT = judgement_score
                        answer_correct_list[query_idx] = is_answer_correct
                        box_match_list[query_idx] = box_match
                        all_match_list[query_idx] = box_match + format_match_list[query_idx]
                        loss_mask_enforce_superior_list[query_idx] = (scoring_wCoT >= loss_mask_enforce_superior_list[query_idx])

                        # box_match_process = max(-2.0, box_match + repetition_penalty_list[query_idx])
                        # box_match_list[query_idx] = box_match_process
                        # all_match_list[query_idx] = box_match_process + format_match_list[query_idx]
                        # ray.logger.info(f"Raw Reward={box_match}, Raw Repetition Penalty={repetition_penalty_list[query_idx]}, Processed Reward={box_match_process}")
                    complex_time = time.time() - complex_start
                    ray.logger.info(f"Complex scoring took {complex_time:.2f}s")

                else:
                    ray.logger.info("Processing idle scoring...")
                    complex_start = time.time()
                    ref_reward_idle = self.reward_model[0].forward.remote(sequences_cpu, num_actions, attention_mask_cpu, packed_seq_lens=packed_seq_lens)
                    ref_reward_idle_list = ray.get(ref_reward_idle)
                    # print("ref_reward_idle_list", ref_reward_idle_list.shape)
                    complex_time = time.time() - complex_start
                    ray.logger.info(f"Complex scoring [IDLE] took {complex_time:.2f}s")

            if (self.reward_model is not None):
                ray.get([self.reward_model[0].empty_cache.remote()])

            # import pdb;pdb.set_trace();
            final_answer_reward = torch.tensor(box_match_list, device=attention_mask.device)
            format_reward = torch.tensor(format_match_list, device=attention_mask.device)
            all_reward = torch.tensor(all_match_list, device=attention_mask.device)
            format_correct_tensor = torch.tensor(format_correct_list, device=attention_mask.device)
            answer_correct_tensor = torch.tensor(answer_correct_list, device=attention_mask.device)
            loss_mask_enforce_superior_list_tensor = torch.tensor(loss_mask_enforce_superior_list, device=attention_mask.device)

            r_refs_answer.append(final_answer_reward)
            r_refs_format.append(format_reward)
            r_refs.append(all_reward)
            all_format_correct_list.append(format_correct_tensor)
            all_answer_correct_list.append(answer_correct_tensor)
            all_loss_mask_enforce_superior_list.append(loss_mask_enforce_superior_list_tensor)

            total_time = time.time() - rule_based_start
            ray.logger.info(f"make_experience reward total completed in {total_time:.2f}s")

        else:
            # remote RM
            for rm in self.remote_rm_url:
                if not self.packing_samples:
                    queries = self.tokenizer.batch_decode(sequences_cpu, skip_special_tokens=False)
                    r = remote_rm_fn_ray.remote(rm, queries=queries)
                    r_refs.append(r)
                else:
                    sequences_list = []
                    offset = 0
                    tokens_list = sequences_cpu.tolist()[0]
                    for length in packed_seq_lens:
                        sequences_list.append(tokens_list[offset : offset + length])
                        offset += length
                    queries = self.tokenizer.batch_decode(sequences_list, skip_special_tokens=False)
                    r = remote_rm_fn_ray.remote(rm, queries=queries)
                    r_refs.append(r)
        
        # import pdb;pdb.set_trace();
        for idx in range(len(model_output_experience_list)):
            if idx % 64 == 0 and self.strategy.is_rank_0():
                model_input_idx, model_output_idx, answer_idx, task_idx = model_output_experience_list[idx]
                printing_str = f"Idx {idx}\n\nInput:\n\n{model_input_idx}\n\n"
                printing_str += "---"*50
                printing_str += f"\nOutput:\n\n{model_output_idx}\n\n"
                printing_str += "---"*50
                printing_str += f"\n\nGround-Truth:\n\n{answer_idx}\n\n"
                printing_str += "---"*50
                printing_str += f"\n\nTask source:\n\n{task_idx}\n\n"
                printing_str += "==="*50
                ray.logger.info(printing_str)

        # log probs
        action_log_probs = self.actor(sequences, num_actions, attention_mask, packed_seq_lens=packed_seq_lens)
        actor_value_rm_time = time.time() - start
        # ray.logger.info("finish cal actor log probs")
        # wait initial/critic/reward model done
        start = time.time()
        # print("action_log_probs", action_log_probs.size())
        # print("base_action_log_probs_ref", base_action_log_probs_ref.size())
        # import pdb; pdb.set_trace();
        ref_values = ray.get([base_action_log_probs_ref, value_ref])
        # ray.logger.info("finish cal init log probs and value")
        wait_time = time.time() - start

        base_action_log_probs, value = ref_values[0], ref_values[1]
        rewards = r_refs
        if base_action_log_probs is not None:
            base_action_log_probs = base_action_log_probs.to(device)
        if value is not None:
            value = value.to(device)
        rewards = [r.to(device) for r in rewards]
        rewards_format = [r.to(device) for r in r_refs_format]
        rewards_answer = [r.to(device) for r in r_refs_answer]
        acc_format = [a.to(device) for a in all_format_correct_list]
        acc_answer = [a.to(device) for a in all_answer_correct_list]
        enforce_superior = [efs.to(device) for efs in all_loss_mask_enforce_superior_list]

        r = self.reward_fn(rewards) if len(rewards) > 0 else rewards[0]
        r_format = self.reward_fn(rewards_format) if len(rewards_format) > 0 else rewards_format[0]
        r_acc = self.reward_fn(rewards_answer) if len(rewards_answer) > 0 else rewards_answer[0]
        # import pdb;pdb.set_trace();
        r_acc_all = torch.cat(rewards_answer, dim=0).unsqueeze(-1) if len(rewards_answer) > 0 else rewards_answer[0].unsqueeze(-1)
        enforce_superior_final = torch.cat(enforce_superior, dim=0).unsqueeze(-1) if len(enforce_superior) > 0 else enforce_superior[0].unsqueeze(-1)
        format_accuracy = torch.stack(acc_format).mean(dim=0) if len(acc_format) > 0 else acc_format[0]
        answer_accuracy = torch.stack(acc_answer).mean(dim=0) if len(acc_answer) > 0 else acc_answer[0]

        # avoid CUDA OOM when colocate models
        if self.strategy.args.colocate_critic_reward and not self.remote_rm_url:
            ray.get([self.reward_model[0].empty_cache.remote()])

        if self.strategy.args.colocate_actor_ref:
            torch.cuda.empty_cache()

        if (self.initial_model is not None) and (not args.use_kl_loss):
            kl = compute_approx_kl(
                action_log_probs,
                base_action_log_probs,
                action_mask=action_mask,
                use_kl_estimator_k3=self.strategy.args.use_kl_estimator_k3,
            )
        else:
            kl = torch.zeros_like(action_log_probs, dtype=action_log_probs.dtype, device=device)

        # ray.logger.info("finish cal kl")
        if not self.packing_samples:
            kl_mean = masked_mean(kl, action_mask, dim=-1)
        else:
            # convert tensor into list of tensors so that it's easier to manipulate
            # within dataset.
            sequences = unpacking_samples(sequences, packed_seq_lens)
            attention_mask = None
            action_log_probs = unpacking_samples(action_log_probs, num_actions)
            if value is not None:
                value = unpacking_samples(value, num_actions)

            kl = unpacking_samples(kl, num_actions)
            kl_mean = torch.tensor([each_kl.mean() for each_kl in kl], device=device)

        # info = {
        #     "kl": kl_mean,
        #     "reward": r,
        #     "response_length": samples.response_length,
        #     "total_length": samples.total_length,
        #     "num_actions": num_actions,
        # }
        # prompts_for_grouping = [[self.tokenizer.encode(prompt)] for prompt in prompts]
        prompts_for_grouping = [[prompt] for prompt in prompts]
        info = {
            "kl": kl_mean,
            "reward": r,
            "response_length": samples.response_length,
            "total_length": samples.total_length,
            "num_actions": num_actions,
            "reward_format": r_format,
            "reward_accuracy": r_acc,
            "format_accuracy": format_accuracy,
            "answer_accuracy": answer_accuracy,
        }
        if self.strategy.args.perf:
            self.perf_stats["actor_value_rm_time"] += actor_value_rm_time
            self.perf_stats["wait_time"] += wait_time

        experience = Experience(
            sequences,
            action_log_probs,
            base_action_log_probs,
            value,
            None,
            None,
            attention_mask,
            action_mask,
            info,
            kl,
            enforce_superior_final,
            r_acc_all,
            prompts_for_grouping
        )
        # print("prompts", prompts)
        # import pdb;pdb.set_trace();
        assert(enforce_superior_final.size(0) == action_log_probs.size(0))
        assert(r_acc_all.size(0) == r_acc_all.size(0))
        self.actor.train()  # reset model state
        return experience

    def _generate_vllm(self, all_prompts: List[str], all_labels, all_tasks, **kwargs) -> List[Samples]:
        from vllm import SamplingParams

        # round-robin load balance
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()

        # Select LLM engines: assign each rank an engine, or cycle through engines if world_size < engine_count
        if len(self.vllm_engines) <= world_size:
            llms = [self.vllm_engines[rank % len(self.vllm_engines)]]
        else:
            llms = self.vllm_engines[rank::world_size]

        args = self.strategy.args
        # print("For debugging:", "qwen" in self.strategy.args.pretrain.lower())
        sampling_params = SamplingParams(
            temperature=kwargs.get("temperature", 1.0),
            top_p=kwargs.get("top_p", 1.0),
            top_k=kwargs.get("top_k", -1),
            max_tokens=kwargs.get("max_new_tokens", 1024),
            min_tokens=kwargs.get("min_new_tokens", 1),
            skip_special_tokens=kwargs.get("skip_special_tokens", False),
            stop=["</s>", "<|im_end|>", "<|endoftext|>"] if "qwen" in self.strategy.args.pretrain.lower() else [],
            stop_token_ids=[151645, 151643] if "qwen" in self.strategy.args.pretrain.lower() else [],
        )

        # Expand prompt list based on the number of samples per prompt
        enforce_superior_CoT = getattr(args, "enforce_superior_CoT", False)
        if enforce_superior_CoT:
            # add samples without reasoning
            all_prompts_list = []
            all_labels_list = []
            all_tasks_list = []
            for prompt, label, task in zip(all_prompts, all_labels, all_tasks):
                if task.startswith("ifeval") or task.startswith("complex"):
                    if prompt.endswith("<think>"):
                        prompt_woCoT = prompt + "\n\n</think>\n"
                    elif prompt.endswith("<think>\n"):
                        prompt_woCoT = prompt + "\n</think>\n"
                    elif prompt.endswith("<think>\n\n"):
                        prompt_woCoT = prompt + "</think>\n"
                    else:
                        prompt_woCoT = prompt + "<think>\n\n</think>\n"

                    assert(args.n_samples_per_prompt % 4 == 0)
                    all_prompts_list.append([prompt] * (args.n_samples_per_prompt * 3 // 4) + [prompt_woCoT] * (args.n_samples_per_prompt // 4))
                    all_labels_list.append([label] * (args.n_samples_per_prompt))
                    all_tasks_list.append([task] * (args.n_samples_per_prompt))
                
                else:
                    all_prompts_list.append([prompt] * args.n_samples_per_prompt)
                    all_labels_list.append([label] * args.n_samples_per_prompt)
                    all_tasks_list.append([task] * args.n_samples_per_prompt)

            all_prompts = sum(all_prompts_list, [])
            all_labels = sum(all_labels_list, [])
            all_tasks = sum(all_tasks_list, [])

        else:
            all_prompts = sum([[prompt] * args.n_samples_per_prompt for prompt in all_prompts], [])
            all_labels = sum([[label] * args.n_samples_per_prompt for label in all_labels], [])
            all_tasks = sum([[task] * args.n_samples_per_prompt for task in all_tasks], [])
        
        all_prompt_token_ids = self.tokenize_fn(all_prompts, self.prompt_max_len, padding=False)["input_ids"]

        # Distribute requests to engines and collect responses to outputs
        refs = []
        batch_size = (len(all_prompt_token_ids) + len(llms) - 1) // len(llms)
        for i, llm in enumerate(llms):
            prompt_token_ids = all_prompt_token_ids[i * batch_size : (i + 1) * batch_size]
            refs.append(
                llm.add_requests.remote(rank, sampling_params=sampling_params, prompt_token_ids=prompt_token_ids)
            )
        ray.get(refs)

        # Make sure all requests are sent.
        torch.distributed.barrier()

        # Retrieve and combine results from all outputs
        all_output_refs = []
        for i, llm in enumerate(llms):
            all_output_refs.append(llm.get_responses.remote(rank))

        # Retrieve and combine results from all outputs
        all_outputs = sum(ray.get(all_output_refs), [])

        samples_list = []
        for i in range(0, len(all_outputs), args.micro_rollout_batch_size):
            outputs = all_outputs[i : i + self.strategy.args.micro_rollout_batch_size]
            prompts = all_prompts[i : i + self.strategy.args.micro_rollout_batch_size]
            labels = all_labels[i : i + self.strategy.args.micro_rollout_batch_size]
            tasks = all_tasks[i : i + self.strategy.args.micro_rollout_batch_size]
            if not self.packing_samples:
                # NOTE: concat all outputs to following format:
                #
                # | [PAD] [PAD] token token token | token token [EOS] [PAD] |
                # | token token token token token | token token [EOS] [PAD] |
                # | [PAD] [PAD] [PAD] token token | token token token [EOS] |
                # |<---------- prompt ----------->|<-------- answer ------->|
                max_input_len, max_output_len = 0, 0
                for output in outputs:
                    max_input_len = max(max_input_len, len(output.prompt_token_ids))
                    max_output_len = max(max_output_len, len(output.outputs[0].token_ids))

                pad_token_id, eos_token_id = self.tokenizer.pad_token_id, self.tokenizer.eos_token_id
                sequences = []
                for output in outputs:
                    # left padding input
                    input_len = len(output.prompt_token_ids)
                    input_ids = [pad_token_id] * (max_input_len - input_len) + list(output.prompt_token_ids)

                    # right padding output
                    output_len = len(output.outputs[0].token_ids)
                    output_ids = list(output.outputs[0].token_ids) + [pad_token_id] * (max_output_len - output_len)

                    if output_ids[output_len - 1] != eos_token_id:
                        output_ids[min(output_len, len(output_ids) - 1)] = eos_token_id

                    # concat input and output
                    sequences.append(input_ids + output_ids)

                sequences = torch.tensor(sequences)
                sequences, attention_mask, action_mask = self.actor.process_sequences(
                    sequences, max_input_len, eos_token_id, pad_token_id
                )
                sequences = sequences.to("cuda")
                attention_mask = attention_mask.to("cuda")
                action_mask = action_mask.to("cuda")
                samples_list.append(
                    SamplesBOX(
                        sequences=sequences,
                        attention_mask=attention_mask,
                        action_mask=action_mask,
                        num_actions=action_mask.size(1),
                        packed_seq_lens=None,
                        response_length=action_mask.float().sum(dim=-1),
                        total_length=attention_mask.float().sum(dim=-1),
                        prompts=prompts,
                        answers=labels,
                        tasks=tasks,
                        generate_kwargs=kwargs,
                    )
                )
            else:
                # NOTE: concat all outputs to following format:
                #
                # | token token token | token token [EOS] | token token token token token | token token [EOS] | token token | token token token [EOS] |
                # |<---  prompt ----->|<---- answer ----->|<---------- prompt ----------->|<----- answer ---->|<- prompt -->|<-------- answer ------->|
                pad_token_id, eos_token_id = self.tokenizer.pad_token_id, self.tokenizer.eos_token_id
                sequences = []
                packed_seq_lens = []
                attention_mask = []
                num_actions = []
                for i, output in enumerate(outputs):
                    input_len = len(output.prompt_token_ids)
                    output_len = len(output.outputs[0].token_ids)
                    packed_seq_lens.append(input_len + output_len)
                    sequences.extend(output.prompt_token_ids + list(output.outputs[0].token_ids))
                    attention_mask.extend([i + 1] * (input_len + output_len))

                    # current_action_mask = [0] * (input_len - 1) + [1] * output_len + [0]
                    # num_actions.append(max(1, sum(current_action_mask)))
                    num_actions.append(max(1, output_len))

                sequences = torch.tensor(sequences, device="cuda").unsqueeze(0)
                attention_mask = torch.tensor(attention_mask, device="cuda").unsqueeze(0)
                action_mask = None
                response_length = torch.tensor(num_actions, device="cuda", dtype=torch.float)
                total_length = torch.tensor(packed_seq_lens, device="cuda", dtype=torch.float)
                samples_list.append(
                    SamplesBOX(
                        sequences=sequences,
                        attention_mask=attention_mask,
                        action_mask=None,
                        num_actions=num_actions,
                        packed_seq_lens=packed_seq_lens,
                        response_length=response_length,
                        total_length=total_length,
                        prompts=prompts,
                        answers=labels,
                        tasks=tasks,
                        generate_kwargs=kwargs         
                    )
                )
        return samples_list

    def flush(self):
        "Ensure all experience has been send to critic"
        if self.critic is not None:
            ray.get(self._ref)
            self._ref = None


class RemoteExperienceMakerPRMBOX(NaiveExperienceMakerPRM800K_BOX):
    def __init__(self, *args, vllm_engines: List = None, packing_samples=False, vllm_engines_fixed: List = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.vllm_engines = vllm_engines
        self.vllm_engines_fixed = vllm_engines_fixed
        self.packing_samples = packing_samples
        if self.custom_reward_func:
            self.custom_reward_func = ray.remote(self.custom_reward_func)

    @torch.no_grad()
    def make_experience_list(self, all_prompts: Union[str, List[str]], all_answers:  Union[str, List[str]], all_tasks, **generate_kwargs) -> List[Experience]:
        if self.strategy.args.perf:
            self.perf_stats = {
                "generate_time": 0,
                "actor_value_rm_time": 0,
                "wait_time": 0,
            }
        experiences = super().make_experience_list(all_prompts, all_answers, all_tasks, **generate_kwargs)
        if self.critic is not None:
            for experience in experiences:
                # send experience to critic
                experience_cpu = deepcopy(experience)
                experience_cpu.to_device("cpu")
                self._ref = self.critic.append.remote(experience_cpu)
        return experiences

    @torch.no_grad()
    def generate_samples(self, all_prompts: List[str], all_answers: List[str], all_tasks, **generate_kwargs) -> List[Samples]:
        """
        Generate samples and return in batches.

        When not using vllm, we will fallback to the default implementation,
        in which actor will be used to generate samples.
        """
        if self.vllm_engines is None:
            return super().generate_samples(all_prompts, all_answers, all_tasks, **generate_kwargs)

        # vLLM generation
        samples = self._generate_vllm(all_prompts, all_labels, all_tasks, **generate_kwargs)

        # vLLM offload when colocate_all_models
        if self.strategy.args.vllm_enable_sleep:
            if torch.distributed.get_rank() == 0:
                refs = []
                for engine in self.vllm_engines:
                    refs.append(engine.sleep.remote())
                ray.get(refs)
        return samples


    @torch.no_grad()
    def make_experience(self, samples: Samples, **generate_kwargs) -> Experience:
        """
        Turn samples into experience by calculating logprobs, values, rewards, and kl divergence.
        """
        args = self.strategy.args
        self.actor.eval()
        device = torch.cuda.current_device()
        # extract values from samples
        sequences = samples.sequences
        attention_mask = samples.attention_mask
        action_mask = samples.action_mask
        num_actions = samples.num_actions
        packed_seq_lens = samples.packed_seq_lens
        answers = samples.answers
        tasks = samples.tasks
        start = time.time()
        sequences_cpu, attention_mask_cpu = (
            sequences.to("cpu"),
            attention_mask.to("cpu"),
        )
        if self.initial_model is not None:
            base_action_log_probs_ref = self.initial_model.forward.remote(
                sequences_cpu, num_actions, attention_mask_cpu, packed_seq_lens=packed_seq_lens
            )

            if args.colocate_actor_ref or args.colocate_all_models:
                ray.get([base_action_log_probs_ref])
                ray.get([self.initial_model.empty_cache.remote()])
        else:
            base_action_log_probs_ref = ray.put(None)

        # values
        if self.critic is not None:
            value_ref = self.critic.forward.remote(
                sequences_cpu, num_actions, attention_mask_cpu, packed_seq_lens=packed_seq_lens
            )
            # avoid CUDA OOM when colocate models
            if self.strategy.args.colocate_critic_reward:
                ray.get([value_ref])
                ray.get([self.critic.empty_cache.remote()])
        else:
            value_ref = ray.put(None)

        if self.strategy.args.colocate_actor_ref:
            ray.get([base_action_log_probs_ref])
            ray.get([self.initial_model.empty_cache.remote()])

        # rewards
        r_refs = []
        prm_refs = []
        # support remote RM API with ray
        if not self.remote_rm_url:
            queries = self.tokenizer.batch_decode(sequences.cpu(), skip_special_tokens=False)
            pattern = r"The final answer is: \\boxed\{(.*?)\}"
            missing_answer_indices = [
            i for i, query in enumerate(queries) if not re.search(pattern, query, re.DOTALL)
            ]
            processed_queries = []
            box_match_list = []
            #math_equal_list = []
            for query, answer in zip(queries, answers):
                #temp_query = query
                query, box_match = preprocess_box_responsev2(query, answer)
                processed_queries.append(query)
                box_match_list.append(box_match)
                
            queries = processed_queries
            
            final_answer_reward = torch.tensor(box_match_list, device=device)
            
            
            reward_sequences, reward_attention_mask = preprocess_prm_reward(queries, self.tokenizer, self.prompt_max_len, **generate_kwargs)
            
            r_refs.append(final_answer_reward)
            for rm in self.reward_model:
                prm_refs.append(rm.forward.remote(reward_sequences, reward_attention_mask, packed_seq_lens=packed_seq_lens))
                
            # print()
        else:
            # remote RM
            for rm in self.remote_rm_url:
                if not self.packing_samples:
                    queries = self.tokenizer.batch_decode(sequences_cpu, skip_special_tokens=False)
                    r = remote_rm_fn_ray.remote(rm, queries=queries)
                    r_refs.append(r)
                else:
                    sequences_list = []
                    offset = 0
                    tokens_list = sequences_cpu.tolist()[0]
                    for length in packed_seq_lens:
                        sequences_list.append(tokens_list[offset : offset + length])
                        offset += length
                    queries = self.tokenizer.batch_decode(sequences_list, skip_special_tokens=False)
                    r = remote_rm_fn_ray.remote(rm, queries=queries)
                    r_refs.append(r)

        # log probs
        action_log_probs = self.actor(sequences, num_actions, attention_mask, packed_seq_lens=packed_seq_lens)
        actor_value_rm_time = time.time() - start

        # wait initial/critic/reward model done
        start = time.time()
        ref_values = ray.get([base_action_log_probs_ref, value_ref] + prm_refs)
        wait_time = time.time() - start

        base_action_log_probs, value, prm_rewards = ref_values[0], ref_values[1], ref_values[2]
        final_answer_reward = r_refs[0]
        base_action_log_probs = base_action_log_probs.to(device)
        if value is not None:
            value = value.to(device)
            
        scores = prm_rewards.softmax(dim=-1)[:, :, 0]
        step_scores = []
        step_tag = '<|reserved_special_token_0|>'
        step_tag_id = self.tokenizer.encode(f"{step_tag}")[-1]
        
        for index, score in enumerate(scores):
            if index in missing_answer_indices:
                # 如果 query 不包含 \n\n# Answer\n\n，设置 step_score 为 0.0
                step_scores.append(torch.tensor(-0.5, device=device))
            else:
                # 否则正常计算 step_score
                step_scores.append(torch.tensor(score[reward_sequences[index] == step_tag_id], device=device))       
        reward_indexs = []
        
        for seq in sequences:
            indices = (seq == step_tag_id).nonzero(as_tuple=True)[0]
            if len(indices) > 0:
                reverse_indices = [- (len(seq) - idx.item()) for idx in indices]
            else:
                reverse_indices = [-1]
            # 使用负索引提取128002的位置
            reward_indexs.append(torch.tensor(reverse_indices, device=device))
        extracted_elements = []
        for seq, negative_indices in zip(sequences, reward_indexs):
            elements = [seq[idx].item() for idx in negative_indices]
            extracted_elements.append(elements)      
                
        # 提取每个张量的最小值，若为空则设置为0.0
        min_values = [step_score.min()-0.5 if step_score.numel() > 0 else torch.tensor(-0.5, device=device) for step_score in step_scores]   
        # 将最小值重新合并为一个新的张量
        min_values_tensor = torch.tensor(min_values, device=device)
        
        r = min_values_tensor.to(device=device)
        step_scores = [step_score - 0.5 if step_score.numel() > 0 else torch.tensor(-0.5, device=device) for step_score in step_scores]
        
        # 将 math_box_list 转换为张量，并移动到与 step_scores 相同的设备
        math_box_tensor = torch.tensor(final_answer_reward, device=device)
        if len(math_box_tensor) == len(step_scores):
            r = r + math_box_tensor
            for i in range(len(math_box_tensor)):
                step_score = step_scores[i]
                add_box = math_box_tensor[i]
                if step_score.dim() == 0:
                    step_scores[i] = step_score + add_box
                else:
                    step_scores[i][-1] += add_box

        # avoid CUDA OOM when colocate models
        if self.strategy.args.colocate_critic_reward and not self.remote_rm_url:
            ray.get([self.reward_model[0].empty_cache.remote()])

        if self.strategy.args.colocate_actor_ref:
            torch.cuda.empty_cache()

        if (self.initial_model is not None) and (not args.use_kl_loss):
            kl = compute_approx_kl(
                action_log_probs,
                base_action_log_probs,
                action_mask=action_mask,
                use_kl_estimator_k3=args.use_kl_estimator_k3,
            )
        else:
            kl = torch.zeros_like(action_log_probs, dtype=action_log_probs.dtype, device=device)

        if not self.packing_samples:
            kl_mean = masked_mean(kl, action_mask, dim=-1)
        else:
            # convert tensor into list of tensors so that it's easier to manipulate
            # within dataset.
            sequences = unpacking_samples(sequences, packed_seq_lens)
            attention_mask = None
            action_log_probs = unpacking_samples(action_log_probs, num_actions)
            if value is not None:
                value = unpacking_samples(value, num_actions)

            kl = unpacking_samples(kl, num_actions)
            kl_mean = torch.tensor([each_kl.mean() for each_kl in kl], device=device)

        for i, tensor in enumerate(reward_indexs):
            if not torch.equal(tensor, torch.tensor([-1], device=device)):
                reward_indexs[i] = torch.cat((tensor[:-2], tensor[-1:]))
            
        info = {
            "kl": kl_mean,
            "reward": r,
            "response_length": samples.response_length,
            "total_length": samples.total_length,
            "num_actions": num_actions,
        }
        if self.strategy.args.perf:
            self.perf_stats["actor_value_rm_time"] += actor_value_rm_time
            self.perf_stats["wait_time"] += wait_time

        experience = Experience(
            sequences,
            action_log_probs,
            base_action_log_probs,
            value,
            None,
            None,
            attention_mask,
            action_mask,
            info,
            kl,
        ), reward_indexs, step_scores
        
        self.actor.train()  # reset model state
        return experience

    def _generate_vllm(self, all_prompts: List[str], all_answers: List[str], all_tasks, **kwargs) -> List[Samples]:
        from vllm import SamplingParams

        # round-robin load balance
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()

        # Select LLM engines: assign each rank an engine, or cycle through engines if world_size < engine_count
        if len(self.vllm_engines) <= world_size:
            llms = [self.vllm_engines[rank % len(self.vllm_engines)]]
        else:
            llms = self.vllm_engines[rank::world_size]

        args = self.strategy.args

        sampling_params = SamplingParams(
            temperature=kwargs.get("temperature", 1.0),
            top_p=kwargs.get("top_p", 1.0),
            top_k=kwargs.get("top_k", -1),
            max_tokens=kwargs.get("max_new_tokens", 1024),
            min_tokens=kwargs.get("min_new_tokens", 1),
            skip_special_tokens=kwargs.get("skip_special_tokens", False),
        )

        # Expand prompt list based on the number of samples per prompt
        all_prompts = sum([[prompt] * args.n_samples_per_prompt for prompt in all_prompts], [])
        all_answers = sum([[answer] * args.n_samples_per_prompt for answer in all_answers], [])
        all_prompt_token_ids = self.tokenize_fn(all_prompts, self.prompt_max_len, padding=False)["input_ids"]

        # Distribute requests to engines and collect responses to outputs
        # all_output_refs = []
        # batch_size = (len(all_prompt_token_ids) + len(llms) - 1) // len(llms)
        # for i, llm in enumerate(llms):
        #     prompt_token_ids = all_prompt_token_ids[i * batch_size : (i + 1) * batch_size]
        #     if prompt_token_ids:
        #         all_output_refs.append(
        #             llm.generate.remote(sampling_params=sampling_params, prompt_token_ids=prompt_token_ids)
        #         )

        # Distribute requests to engines and collect responses to outputs
        refs = []
        batch_size = (len(all_prompt_token_ids) + len(llms) - 1) // len(llms)
        for i, llm in enumerate(llms):
            prompt_token_ids = all_prompt_token_ids[i * batch_size : (i + 1) * batch_size]
            refs.append(
                llm.add_requests.remote(rank, sampling_params=sampling_params, prompt_token_ids=prompt_token_ids)
            )
        ray.get(refs)

        # Make sure all requests are sent.
        torch.distributed.barrier()

        # Retrieve and combine results from all outputs
        all_output_refs = []
        for i, llm in enumerate(llms):
            all_output_refs.append(llm.get_responses.remote(rank))

        # Retrieve and combine results from all outputs
        all_outputs = sum(ray.get(all_output_refs), [])

        samples_list = []
        for i in range(0, len(all_outputs), args.micro_rollout_batch_size):
            outputs = all_outputs[i : i + self.strategy.args.micro_rollout_batch_size]
            answers = all_answers[i : i + self.strategy.args.micro_rollout_batch_size]
            if not self.packing_samples:
                # NOTE: concat all outputs to following format:
                #
                # | [PAD] [PAD] token token token | token token [EOS] [PAD] |
                # | token token token token token | token token [EOS] [PAD] |
                # | [PAD] [PAD] [PAD] token token | token token token [EOS] |
                # |<---------- prompt ----------->|<-------- answer ------->|
                max_input_len, max_output_len = 0, 0
                for output in outputs:
                    max_input_len = max(max_input_len, len(output.prompt_token_ids))
                    max_output_len = max(max_output_len, len(output.outputs[0].token_ids))

                pad_token_id, eos_token_id = self.tokenizer.pad_token_id, self.tokenizer.eos_token_id
                sequences = []
                for output in outputs:
                    # left padding input
                    input_len = len(output.prompt_token_ids)
                    input_ids = [pad_token_id] * (max_input_len - input_len) + list(output.prompt_token_ids)

                    # right padding output
                    output_len = len(output.outputs[0].token_ids)
                    output_ids = list(output.outputs[0].token_ids) + [pad_token_id] * (max_output_len - output_len)

                    if output_ids[output_len - 1] != eos_token_id:
                        output_ids[min(output_len, len(output_ids) - 1)] = eos_token_id

                    # concat input and output
                    sequences.append(input_ids + output_ids)

                sequences = torch.tensor(sequences)
                sequences, attention_mask, action_mask = self.actor.process_sequences(
                    sequences, max_input_len, eos_token_id, pad_token_id
                )
                sequences = sequences.to("cuda")
                attention_mask = attention_mask.to("cuda")
                action_mask = action_mask.to("cuda")
                samples_list.append(
                    SamplesBOX(
                        sequences=sequences,
                        attention_mask=attention_mask,
                        action_mask=action_mask,
                        num_actions=action_mask.size(1),
                        packed_seq_lens=None,
                        response_length=action_mask.float().sum(dim=-1),
                        total_length=attention_mask.float().sum(dim=-1),
                        answers=answers,
                    )
                )
            else:
                # NOTE: concat all outputs to following format:
                #
                # | token token token | token token [EOS] | token token token token token | token token [EOS] | token token | token token token [EOS] |
                # |<---  prompt ----->|<---- answer ----->|<---------- prompt ----------->|<----- answer ---->|<- prompt -->|<-------- answer ------->|
                pad_token_id, eos_token_id = self.tokenizer.pad_token_id, self.tokenizer.eos_token_id
                sequences = []
                packed_seq_lens = []
                attention_mask = []
                num_actions = []
                for i, output in enumerate(outputs):
                    input_len = len(output.prompt_token_ids)
                    output_len = len(output.outputs[0].token_ids)
                    packed_seq_lens.append(input_len + output_len)
                    sequences.extend(output.prompt_token_ids + list(output.outputs[0].token_ids))
                    attention_mask.extend([i + 1] * (input_len + output_len))

                    # current_action_mask = [0] * (input_len - 1) + [1] * output_len + [0]
                    # num_actions.append(max(1, sum(current_action_mask)))
                    num_actions.append(max(1, output_len))

                sequences = torch.tensor(sequences, device="cuda").unsqueeze(0)
                attention_mask = torch.tensor(attention_mask, device="cuda").unsqueeze(0)
                action_mask = None
                response_length = torch.tensor(num_actions, device="cuda", dtype=torch.float)
                total_length = torch.tensor(packed_seq_lens, device="cuda", dtype=torch.float)
                samples_list.append(
                    SamplesBOX(
                        sequences=sequences,
                        attention_mask=attention_mask,
                        action_mask=None,
                        num_actions=num_actions,
                        packed_seq_lens=packed_seq_lens,
                        response_length=response_length,
                        total_length=total_length,
                        answers=answers,
                    )
                )
        return samples_list

    def flush(self):
        "Ensure all experience has been send to critic"
        if self.critic is not None:
            ray.get(self._ref)
            self._ref = None
