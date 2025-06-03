# coding=utf-8
# Copyright 2023 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Binary of evaluating instruction following. See README.md."""

import collections
import dataclasses
import json
import os
from typing import Dict, Optional, Sequence, Union, List
from absl import app
from absl import flags
from absl import logging
from . import instructions_registry
from tqdm import tqdm
import re
from typing import Dict, Tuple, Optional
import json
from openrlhf.trainer.ppo_utils.kk import extract_solution, validate_response_structure



def test_instruction_following_strict(
    prompt, response, instruction_list, instruction_kwargs,
):
    """Tests response to see if instrutions are followed."""
    is_following_list = []

    for index, instruction_id in enumerate(instruction_list):
        instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
        instruction = instruction_cls(instruction_id)
        instruction.build_description(**instruction_kwargs[index])
        args = instruction.get_instruction_args()
        if args and "prompt" in args:
            instruction.build_description(prompt=prompt)
        if response is None:
            response = ""
        if response.strip() and instruction.check_following(response):
            is_following_list.append(True)
        else:
            is_following_list.append(False)
    
    return is_following_list




def compute_ifeval_score(prompt, response, answer_json, template, ignore_format=False, format_reward=1.0, answer_reward=2.0):
    # Compute the ifeval score
    instruction_list = answer_json["instruction_id_list"]
    instruction_kwargs = answer_json["kwargs"]
    # print(f"[Ground Truth] InstructionList: {instruction_list}")
    # print(f"[Ground Truth] InstructionKWArgs: {instruction_kwargs}")
    if "follow_score_qwen2.5-7B-Instruct" in answer_json:
        # TODO: replace the off-line scoring with the on-line scoring
        reference_scoring_woCoT = answer_json["follow_score_qwen2.5-7B-Instruct"]
    else:
        # if not existing simply ignore it
        reference_scoring_woCoT = 0

    # Extract model answer
    if ignore_format:
        format_correct = True
        answer_text = response
        processed_str = response
        # print("---Ignore Formatting---")

    else:
        answer_text, think_text, processed_str = extract_solution(response, template=template)
        # Validate response structure
        format_correct = validate_response_structure(processed_str, template=template)
    
    # print(f"\n[Model Response]\n{processed_str}")
    # print(f"\n  Format validation: {'PASS' if format_correct else 'FAIL'}")

    is_answer_correct = 0
    if format_correct and answer_text and (think_text and think_text.lower() != "reasoning process here"):

        is_following_list = test_instruction_following_strict(prompt, answer_text, instruction_list, instruction_kwargs)
        # Validate answer content
        answer_score = 0.0
        is_answer_superior_to_answer_woCoT = (sum(is_following_list) >= reference_scoring_woCoT)

        if sum(is_following_list) == len(is_following_list):
            answer_score = answer_reward
            is_answer_correct = 1
            # print("  Content validation: FULL Instruction Alignment")
        
        else:
            if sum(is_following_list) == 0:
                answer_score = -abs(answer_reward)
                # print("  Content validation: FULL MisAlignment")
            else:
                answer_score = float(sum(is_following_list)/len(is_following_list))
                is_answer_correct = float(sum(is_following_list)/len(is_following_list))
                # print("  Content validation: Part MisAlignment")
    
    else:
        answer_score = -abs(answer_reward)
        # For answers without valid formatting we simply ignore it
        is_answer_superior_to_answer_woCoT = True
        # print("\n[Content Validation] Skipped due to format errors or missing answer")

    return answer_score, is_answer_correct, is_answer_superior_to_answer_woCoT



def test_instruction_following_loose(
    prompt, response, instruction_list, instruction_kwargs):
    """Tests response for an upper bound for following instructions."""
    r = response.split("\n")
    response_remove_first = "\n".join(r[1:]).strip()
    response_remove_last = "\n".join(r[:-1]).strip()
    response_remove_both = "\n".join(r[1:-1]).strip()
    revised_response = response.replace("*", "")
    revised_response_remove_first = response_remove_first.replace("*", "")
    revised_response_remove_last = response_remove_last.replace("*", "")
    revised_response_remove_both = response_remove_both.replace("*", "")
    all_responses = [
        response,
        revised_response,
        response_remove_first,
        response_remove_last,
        response_remove_both,
        revised_response_remove_first,
        revised_response_remove_last,
        revised_response_remove_both,
    ]
    is_following_list = []

    for index, instruction_id in enumerate(instruction_list):
        instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
        instruction = instruction_cls(instruction_id)

        instruction.build_description(**instruction_kwargs[index])
        args = instruction.get_instruction_args()
        if args and "prompt" in args:
            instruction.build_description(prompt=prompt)

        is_following = False
        for r in all_responses:
            if r.strip() and instruction.check_following(r):
                is_following = True
                break

        is_following_list.append(is_following)
    
    return is_following_list




if __name__ == "__main__":

    input_jsonl_filename = "/apdcephfs_cq8/share_2992827/shennong_5/yuleiqin/youtu-reid/lm-eval-business/COMPLEX_INSTRUCTIONS/FollowComplexInstruction/evaluations_gen/Qwen2.5-72B-INT8/follow_complex_instruction_final.jsonl"
    save_jsonl_filename = input_jsonl_filename.replace(".jsonl", "_valid.jsonl")
    assert(save_jsonl_filename != input_jsonl_filename)
    info_list = []
    with open(input_jsonl_filename, "r") as fr:
        for line in fr:
            info = json.loads(line)
            info_list.append(info)
    
    info_list_valid = []
    for info in tqdm(info_list):
        prompt = info["conversations"][0]["content"]
        response = info["conversations"][1]["content"]
        instruction_list = info["instruction_id_list"]
        instruction_kwargs = info["kwargs"]
        is_following_list = test_instruction_following_strict(prompt, response, instruction_list, instruction_kwargs)
        is_following_list_loose = test_instruction_following_loose(prompt, response, instruction_list, instruction_kwargs)
        is_continue = False
        for is_following_item, is_following_loose_item in zip(is_following_list, is_following_list_loose):
            if is_following_item:
                if not (is_following_loose_item):
                    is_continue = True
                    break
        if is_continue:
            continue
        
        info_list_valid.append(info)
    
    with open(save_jsonl_filename, "w") as fw:
        for info in info_list:
            fw.write(json.dumps(info, ensure_ascii=False)+"\n")




