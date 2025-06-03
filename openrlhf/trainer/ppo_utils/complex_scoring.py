import math
import os
import os.path
from abc import ABC
from typing import Any, Callable, Dict, List, Optional, Union
import json
import ray
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from openrlhf.models.utils import compute_approx_kl, compute_reward, masked_mean, unpacking_samples
from openrlhf.models.utils import log_probs_from_logits, reset_position_ids
from openrlhf.trainer.ppo_utils.kk import extract_solution, validate_response_structure
import math
from copy import deepcopy


SYS_MSG = """Based on the provided <Input> and <Generated Text>, answer the ensuing <Question> with either a "YES(是)" or "NO(否)" choice. Your selection should be based on your judgment as well as the following rules:

- YES(是): Select 'YES(是)' if the generated text entirely fulfills the condition specified in the question. However, note that even minor inaccuracies exclude the text from receiving a 'YES(是)' rating.
As an illustration, consider a question that asks \"Does each sentence in the generated text use a second person narration?\". Even if only one sentence does not use the second person, the answer should NOT be 'YES(是)'.
To qualify for a 'YES(是)' rating, the generated text must be entirely accurate and relevant to the question.

- NO(否): Opt for 'NO(否)' if the generated text fails to meet the question's requirements or provides no information that could be utilized to answer the question.
For instance, if the question asks "Is the second sentence in the generated text a compound sentence?" and the generated text only has one sentence, it offers no relevant information to answer the question. Consequently, the answer should be 'NO(否)'."""



def shrink_system_prompt(system_content):
    SYS_COT_END = """between one <answer> tag and one </answer> tag."""
    SYS_COT_END2 = """<think> reasoning process here </think><answer> answer here </answer>."""
    SYS_SPLIT = "Here is the task definition:\n\n"
    
    system_prompt_default = deepcopy(system_content)
    if SYS_SPLIT in system_prompt_default:
        system_prompt_default = system_prompt_default[system_prompt_default.find(SYS_SPLIT)+len(SYS_SPLIT):]

    if SYS_COT_END in system_prompt_default:
        system_prompt_default = system_prompt_default[system_prompt_default.find(SYS_COT_END)+len(SYS_COT_END):]

    if SYS_COT_END2 in system_prompt_default:
        system_prompt_default = system_prompt_default[system_prompt_default.find(SYS_COT_END2)+len(SYS_COT_END2):]

    return system_prompt_default


def extract_system_user_content_from_model_input(model_input, template="qwen"):
    conversations = []
    
    if template == "qwen":
        assert("<|im_start|>" in model_input)
        assert("<|im_end|>" in model_input)
        text_splits = model_input.split("<|im_start|>")
        text_splits = [item.strip() for item in text_splits if len(item.strip())]
        for text_split in text_splits:
            role = text_split[:text_split.find("\n")]
            if "<|im_end|>" in text_split:
                content = (text_split[text_split.find("\n"):text_split.rfind("<|im_end|>")]).strip()
            else:
                content = (text_split[text_split.find("\n"):]).strip()

            if len(role) and len(content) and role in ["system", "user"]:
                if role == "system":
                    sys_content = shrink_system_prompt(content)
                    if len(sys_content):
                        conversations.append({"role":str(role), "content":sys_content})
                else:
                    conversations.append({"role":str(role), "content":str(content)})

    elif template == "deepseek":
        assert("<｜begin▁of▁sentence｜>" in model_input)
        assert("<｜User｜>" in model_input)
        system_prompt = (model_input[model_input.find("<｜begin▁of▁sentence｜>")+len("<｜begin▁of▁sentence｜>"):model_input.find("<｜User｜>")]).strip()
        text_splits = (model_input[model_input.find("<｜User｜>"):]).split("<｜User｜>")
        if len(system_prompt):
            sys_content = shrink_system_prompt(system_prompt)
            if len(sys_content):
                conversations.append({"role":"system", "content":sys_content})

        for text_split in text_splits:
            if len(text_split.strip()):
                if "<｜Assistant｜>" in text_split:
                    text_split_item = (text_split[:text_split.find("<｜Assistant｜>")]).strip()
                else:
                    text_split_item = text_split.strip()
                if len(text_split_item):
                    conversations.append({"role":"user", "content":text_split_item})
    
    elif template == "llama3":
        assert("<|eot_id|>" in model_input)
        assert("<|begin_of_text|>" in model_input)
        text_splits = model_input.split("<|eot_id|>")
        text_splits = [item.strip() for item in text_splits if len(item.strip())]
        for text_split in text_splits:
            if "<|start_header_id|>" in text_split and "<|end_header_id|>" in text_split:
                role = text_split[text_split.find("<|start_header_id|>")+len("<|start_header_id|>"):text_split.find("<|end_header_id|>")]
                content = text_split[text_split.find("<|end_header_id|>")+len("<|end_header_id|>"):]
                if len(role) and len(content) and role in ["system", "user"]:
                    if role == "system":
                        sys_content = shrink_system_prompt(content)
                        if len(sys_content):
                            conversations.append({"role":str(role), "content":sys_content})
                    else:
                        conversations.append({"role":str(role), "content":str(content)})
    
    elif template == "mistral":
        assert("[INST]" in model_input)
        assert("[/INST]" in model_input)
        text_splits = model_input.split("[INST]")
        text_splits = [item.strip() for item in text_splits if len(item.strip())]
        for text_split in text_splits:
            if "[/INST]" in text_split:
                user_content = text_split[:text_split.find("[/INST]")]
                system_content = shrink_system_prompt(user_content)
                if system_content != user_content and len(system_content):
                    conversations.append({"role":"system", "content":system_content})
                else:
                    # only contains user content
                    conversations.append({"role":"user", "content":user_content})
    
    elif template == "base":
        assert("User:" in model_input)
        if "System:" in model_input:
            system_content = model_input[model_input.find("System:")+len("System:"):model_input.find("User:")]
            conversations.append({"role":"system", "content":shrink_system_prompt(system_content)})
            model_input = model_input[model_input.find("User:"):]

        text_splits = model_input.split("User:")
        text_splits = [item.strip() for item in text_splits if len(item.strip())]
        for text_split in text_splits:
            if "Assistant:" in text_split: 
                user_content = text_split[:text_split.find("Assistant:")]
                conversations.append({"role":"user", "content":user_content})

    else:
        raise NotImplementedError(f"Unsupported template: {template}")

    system_prompt_default = conversations[0]["content"] if conversations[0]["role"] == "system" else ""
    if conversations[-1]["role"] == "user":
        model_input_user_str = (system_prompt_default + "\n\n" + conversations[-1]["content"]).strip()
    else:
        model_input_user_str = system_prompt_default
    return model_input_user_str





def wrapping_prompts(model_input, answer_text, scoring_questions, template="qwen"):
    model_input_user_str = extract_system_user_content_from_model_input(model_input, template)
    wrapping_content_list = []
    wrapping_pairs_list = []
    for scoring_question in scoring_questions:
        wrapping_content = f"{SYS_MSG}\n\n\nInput:\n\n{model_input_user_str}\n\n\nGenerated Text:\n\n{answer_text}\n\n\nQuestion:\n\n{scoring_question}"
        wrapping_content_list.append([
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role":"user",
                "content":wrapping_content,
            },
        ])
        wrapping_pairs_list.append([
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role":"user",
                "content":wrapping_content,
            },
            {
                "role":"assistant",
                "content":"YES(是)"
            }]
        )
        wrapping_pairs_list.append([
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role":"user",
                "content":wrapping_content,
            },
            {
                "role":"assistant",
                "content":"NO(否)"
            }]
        )
    assert(2*len(wrapping_content_list) == len(wrapping_pairs_list))
    assert(len(wrapping_pairs_list) % 2 == 0)
    return wrapping_content_list, wrapping_pairs_list




def tokenize_fn(tokenizer, texts, max_length, padding=True, device=None, apply_chat_template=False,\
    add_generation_prompt=False, add_special_tokens=False, continue_final_message=False, padding_side="right"):
    if apply_chat_template:
        if add_generation_prompt:
            texts = [
                tokenizer.apply_chat_template(message, tokenize=False,\
                    add_generation_prompt=add_generation_prompt,\
                        add_special_tokens=add_special_tokens) for message in texts]
        else:
            texts = [
                tokenizer.apply_chat_template(message, tokenize=False,\
                        add_special_tokens=add_special_tokens,\
                            continue_final_message=continue_final_message) for message in texts]
        # print(f"Texts after applying chat template: {texts}")
    
    text_input_ids_lengths = []
    for text in texts:
        text_input_ids_lengths.append(len(tokenizer.encode(text)))

    if not padding:
        # when padding is False, return tokenized texts as list
        return tokenizer(
            texts,
            add_special_tokens=False,
            max_length=max_length,
            return_attention_mask=True,
            truncation=True,
        )
    batch = tokenizer(
        texts,
        return_tensors="pt",
        add_special_tokens=False,
        max_length=max_length,
        return_attention_mask=True,
        padding=True,
        truncation=True,
        padding_side=padding_side
    )
    return {k: v.to(device) for k, v in batch.items()}, text_input_ids_lengths




def split_by_num_process(items_to_split, N_threads=256):
    """splits the items into N subsequences evenly"""
    num_each_split = len(items_to_split) // N_threads
    # remain_split = len(items_to_split) % N_threads
    nums_split = [[] for _ in range(N_threads)]
    for idx, item_to_split in enumerate(items_to_split):
        if num_each_split != 0:
            idx_split = idx // num_each_split
            if idx_split >= N_threads:
                idx_split = idx % N_threads
        else:
            idx_split = idx % N_threads
        nums_split[idx_split].append(item_to_split)
    nums_split = [num_split for num_split in nums_split if len(num_split) > 0]
    return nums_split




def prepare_complex_judgement(model_input, model_output, answer_json, tokenizer, ignore_format=False, format_reward=1.0, answer_reward=2.0, template="qwen"):

    if ignore_format:
        format_correct = True
        answer_text = model_output
        processed_str = model_output
        # print("---Ignore Formatting---")

    else:
        answer_text, think_text, processed_str = extract_solution(model_output, template=template)
        # Validate response structure
        format_correct = validate_response_structure(processed_str, template=template)

    # print(f"\n[Model Response]\n{processed_str}")
    format_score = format_reward if format_correct else -abs(format_reward)
    # print(f"\n  Format validation: {'PASS' if format_correct else 'FAIL'}")
    # print(f"  Format score: {format_score}")
    # Validate answer content
    answer_score = 0.0
    is_answer_correct = 0
    scoring_questions = answer_json["scoring_questions"]

    if format_correct and answer_text and (think_text and think_text != "reasoning process here"):
        wrapping_content_list_all, wrapping_pairs_list_all = wrapping_prompts(model_input, answer_text, scoring_questions, template)
        scoring_results = []
        num_actions_yes = len(tokenizer.encode("YES(是)"))
        num_action_no = len(tokenizer.encode("NO(否)"))
        num_actions = max(1, min(num_actions_yes, num_action_no))
        answer_score = -100
        is_answer_correct = -100

    else:
        answer_score = -abs(answer_reward)
        # print("\n[Content Validation] Skipped due to format errors or missing answer")
        wrapping_content_list_all = []
        wrapping_pairs_list_all = []
        num_actions = 1
        scoring_questions = []

    return float(answer_score), float(is_answer_correct), wrapping_content_list_all, wrapping_pairs_list_all, num_actions, scoring_questions




def compute_judgement_score_preprocess(complex_scoring_candidates, tokenizer, prompt_max_len):
    """
    input:
        complex_scoring_candidates: [[query_idx, num_actions, wrapping_content_list_all, wrapping_pairs_list_all, scoring_questions],...]
    return:
        judgement_scores: [[query_idx, box_match, is_answer_correct],...]
    """

    num_actions = max(1, min([item[1] for item in complex_scoring_candidates]))
    # print(" num_actions", num_actions)

    indices = []
    wrapping_content_list_all_list = []
    wrapping_pairs_list_all_list = []
    for complex_scoring_candidate in complex_scoring_candidates:
        query_idx = complex_scoring_candidate[0]
        wrapping_content_list_all = complex_scoring_candidate[2]
        wrapping_pairs_list_all = complex_scoring_candidate[3]
        scoring_questions = complex_scoring_candidate[4]
        wrapping_content_list_all_start_indice = len(wrapping_content_list_all_list)
        wrapping_content_list_all_end_indice = wrapping_content_list_all_start_indice + len(wrapping_content_list_all)
        wrapping_pairs_list_all_start_indice = len(wrapping_pairs_list_all_list)
        wrapping_pairs_list_all_end_indice = wrapping_pairs_list_all_start_indice + len(wrapping_pairs_list_all)
        wrapping_content_list_all_list += wrapping_content_list_all
        wrapping_pairs_list_all_list += wrapping_pairs_list_all
        indices.append([query_idx, wrapping_content_list_all_start_indice, wrapping_content_list_all_end_indice,\
            wrapping_pairs_list_all_start_indice, wrapping_pairs_list_all_end_indice, scoring_questions])
    
    prompts_scoring_inputs, _ = tokenize_fn(tokenizer, wrapping_pairs_list_all_list, prompt_max_len,\
        padding=True, device="cpu", apply_chat_template=True, add_generation_prompt=False,\
            continue_final_message=True, padding_side="left")
    return prompts_scoring_inputs, indices, wrapping_content_list_all_list, wrapping_pairs_list_all_list, num_actions




def compute_judgement_score_postprocess(prompts_scoring_inputs, indices, wrapping_content_list_all_list, wrapping_pairs_list_all_list,\
    judge_base_action_log_probs_ref_list_all, answer_reward=2.0):
    """
    input:
        complex_scoring_candidates: [[query_idx, num_actions, wrapping_content_list_all, wrapping_pairs_list_all, scoring_questions],...]
    return:
        judgement_scores: [[query_idx, box_match, is_answer_correct],...]
    """
    judgement_scores = []
    for indice in indices:
        query_idx = indice[0]
        wrapping_content_list_all = wrapping_content_list_all_list[indice[1]:indice[2]]
        wrapping_pairs_list_all = wrapping_pairs_list_all_list[indice[3]:indice[4]]
        judge_base_action_log_probs_ref_list = judge_base_action_log_probs_ref_list_all[indice[3]:indice[4]]
        scoring_questions = indice[5]
        scoring_results = []
        judgement_str = ""
        is_response_nonsense = False

        for batch_idx in range(len(wrapping_content_list_all)):
            judge_log_prob_yes = sum((judge_base_action_log_probs_ref_list[2*batch_idx]).to("cpu"))
            judge_log_prob_no = sum((judge_base_action_log_probs_ref_list[2*batch_idx+1]).to("cpu"))
            # print("judge_log_prob_yes", judge_log_prob_yes)
            # print("judge_log_prob_no", judge_log_prob_no)
            scoring_question = scoring_questions[batch_idx]
            if judge_log_prob_yes > judge_log_prob_no:
                judgement_str += f" {scoring_question}: True ({judge_log_prob_yes} > {judge_log_prob_no})\n"
                scoring_results.append(True)
            else:
                if scoring_question == "是否语义连贯通顺，且不包含注释？(Is it semantically coherent, consistent and fluent, and without any annotations or postscripts?)":
                    is_response_nonsense = True
                    judgement_str += f" Fatal ERROR!!!{scoring_question}: False ({judge_log_prob_yes} <= {judge_log_prob_no})\n"
                else:
                    judgement_str += f" {scoring_question}: False ({judge_log_prob_yes} <= {judge_log_prob_no})\n"
                scoring_results.append(False)
        # import pdb;pdb.set_trace();
        print(f"  Content judgement:\n{judgement_str}")
        # import pdb;pdb.set_trace();
        is_answer_correct = 0
        if sum(scoring_results) == len(scoring_results):
            answer_score = answer_reward
            is_answer_correct = 1
            # print("  Content validation: FULL Instruction Alignment")
        
        else:
            if sum(scoring_results) == 0:
                ## 如果is_response_nonsense放在这里直接惩罚可能过于严厉
                answer_score = -abs(answer_reward)
                # print("  Content validation: FULL MisAlignment")
            else:
                answer_score = float(sum(scoring_results)/len(scoring_results))
                is_answer_correct = float(sum(scoring_results)/len(scoring_results))
                # print("  Content validation: Part MisAlignment")

        judgement_scores.append([query_idx, float(answer_score), float(is_answer_correct), sum(scoring_results)])
    return judgement_scores





if __name__ == "__main__":
    # print()
    # input_json_path = "/cfs/yuleiqin/code/simpleRL-reason/train/data/instruction_following/complex_instruction_collection/complex_R1_filtered/Qwen_template_is-ifeval=2_is-woFormat=False/CFBench_overall_R1_filtered.jsonl"
    # with open(input_json_path, "r") as fr:
    #     for line in fr:
    #         info = json.loads(line)
    #         model_input = info["input"]
    #         model_output = info["answer"]
    #         answer_json = json.loads(info["target"])
    #         scoring_prompts = wrapping_prompts(model_input, model_output, answer_json)
    #         print("scoring_prompts", scoring_prompts)
    #         # import pdb;pdb.set_trace();


    model_input_template = [
        ["qwen", """<|im_start|>system\nYou are a helpful assistant. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>.<|im_end|>\n<|im_start|>user\n你是IT解决方案专家,作为《1 多点备份和多活备份》方案的章节,请完整描述' 多点备份的数据同步和一致性保证机制',要求内容除空格之外在1800字以上,要求内容专业、数据充分、表达清晰、书面化、学术化,不能出现AI,ChatGPT和内容生成的描述,不能出现读者和提问等措辞,不能包含非常抱歉等措辞,不要体现你的身份信息,符合中国网络安全要求\n使用关键词：同步机制. \n首先逐字重复指令，不做任何改变，然后给出你的回答（1. 在重复指令前不要说任何词或字符；2. 你需要重复的指令不包括这句话）<|im_end|>\n<|im_start|>assistant\n"""],
        ["deepseek", """<｜begin▁of▁sentence｜>You are a helpful assistant. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>.<｜User｜>chrome浏览器同步已暂停的原因回应应该包含关键词"同步". 首先重复上面的句子...不要改变任何词，只是重复。在重复句子之前不要做出回答。<｜Assistant｜>"""],
        ["llama3", """<|begin_of_text|><|start_header_id|>system<|end_header_id|>Cutting Knowledge Date: December 2023You are a helpful assistant. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>.<|eot_id|><|start_header_id|>user<|end_header_id|>新时代青年的历史责任回应应该包含关键词"社会进步".先重复上面的指令，然后再回应指令<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""],
        ["mistral", """<s>[INST]You are a helpful assistant. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>.\n\nchrome浏览器同步已暂停的原因\n回应应该包含关键词\"同步\". \n首先重复上面的句子...不要改变任何词，只是重复。在重复句子之前不要做出回答。[/INST]"""],
    ]


    for pair in model_input_template:
        template_name, model_input_str = pair
        print(f"template_name={template_name}\n\n")
        print(extract_system_user_content_from_model_input(model_input_str, template=template_name))
        print("==="*50)

