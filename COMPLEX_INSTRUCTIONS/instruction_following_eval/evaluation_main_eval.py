# coding=utf-8
# Copyright 2024 The Google Research Authors.
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
from tqdm import tqdm
from absl import app
from absl import flags
from absl import logging
from glob import glob
import numpy as np

# from instruction_following_eval import instructions_registry
import instructions_registry


_INPUT_DATA = flags.DEFINE_string(
    "input_data", None, "path to input data", # required=True
)

_INPUT_RESPONSE_DATA = flags.DEFINE_string(
    "input_response_data", None, "path to input response data",# required=False
)

_OUTPUT_DIR = flags.DEFINE_string(
    "output_dir",
    "",
    "Output directory for inference and eval results.",
    required=False,
)

RESULTS_ROOT = "/apdcephfs_cq8/share_2992827/shennong_5/yuleiqin/youtu-reid/lm-eval-business/AAA_Results_Combined"
os.makedirs(RESULTS_ROOT, exist_ok=True)


@dataclasses.dataclass
class InputExample:
  key: int
  instruction_id_list: List[str]
  prompt: str
  kwargs: List[Dict[str, Optional[Union[str, int]]]]


@dataclasses.dataclass
class OutputExample:
  instruction_id_list: List[str]
  prompt: str
  response: str
  follow_all_instructions: bool
  follow_instruction_list: List[bool]


def read_prompt_list(input_jsonl_filename):
  """Read inputs from jsonl."""
  inputs = []
  with open(input_jsonl_filename, "r") as f:
    for l in f:
      example = json.loads(l)
      inputs.append(
          InputExample(key=example["line_idx"], # example["key"],
                       instruction_id_list=example["instruction_id_list"],
                       prompt=example["conversations"][0]["content"], #example["prompt"],
                       kwargs=example["kwargs"]))
  return inputs


def write_outputs(output_jsonl_filename, outputs):
  """Writes outputs to jsonl."""
  assert outputs
  with open(output_jsonl_filename, "w") as f:
    for o in outputs:
      f.write(
          json.dumps(
              {
                  attr_name: o.__getattribute__(attr_name)
                  for attr_name in [
                      name for name in dir(o) if not name.startswith("_")
                  ]
              }
          )
      )
      f.write("\n")


def test_instruction_following_strict(
    inp,
    prompt_to_response,
):
  """Tests response to see if instrutions are followed."""
  response = prompt_to_response[inp.prompt]
  instruction_list = inp.instruction_id_list
  is_following_list = []

  for index, instruction_id in enumerate(instruction_list):
    instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
    instruction = instruction_cls(instruction_id)

    instruction.build_description(**inp.kwargs[index])
    args = instruction.get_instruction_args()
    if args and "prompt" in args:
      instruction.build_description(prompt=inp.prompt)

    if response.strip() and instruction.check_following(response):
      is_following_list.append(True)
    else:
      is_following_list.append(False)

  return OutputExample(
      instruction_id_list=inp.instruction_id_list,
      prompt=inp.prompt,
      response=response,
      follow_all_instructions=all(is_following_list),
      follow_instruction_list=is_following_list,
  )


def test_instruction_following_loose(
    inp,
    prompt_to_response,
):
  """Tests response for an upper bound for following instructions."""
  response = prompt_to_response[inp.prompt]
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
  instruction_list = inp.instruction_id_list
  is_following_list = []


  for index, instruction_id in enumerate(instruction_list):
    instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
    instruction = instruction_cls(instruction_id)

    instruction.build_description(**inp.kwargs[index])
    args = instruction.get_instruction_args()
    if args and "prompt" in args:
      instruction.build_description(prompt=inp.prompt)

    is_following = False
    for r in all_responses:
      if r.strip() and instruction.check_following(r):
        is_following = True
        break

    is_following_list.append(is_following)

  return OutputExample(
      instruction_id_list=inp.instruction_id_list,
      prompt=inp.prompt,
      response=response,
      follow_all_instructions=all(is_following_list),
      follow_instruction_list=is_following_list,
  )


def read_prompt_to_response_dict(input_jsonl_filename):
  """Creates dictionary matching prompt and response."""
  return_dict = {}
  with open(input_jsonl_filename, "r") as f:
    for l in f:
      example = json.loads(l)
      # return_dict[example["prompt"]] = example["response"]
      # import pdb;pdb.set_trace();
      return_dict[example["conversations"][0]["content"]] = example["conversations"][1]["content"]
      # import pdb;pdb.set_trace();

  return return_dict


def print_report(outputs, output_file_name_log):
  """Prints a report on accuracy scores."""

  prompt_total = 0
  prompt_correct = 0
  instruction_total = 0
  instruction_correct = 0

  tier0_total = collections.defaultdict(int)
  tier0_correct = collections.defaultdict(int)

  tier1_total = collections.defaultdict(int)
  tier1_correct = collections.defaultdict(int)

  for example in outputs:
    follow_instruction_list = example.follow_instruction_list
    instruction_id_list = example.instruction_id_list

    prompt_total += 1
    if all(follow_instruction_list):
      prompt_correct += 1

    instruction_total += len(instruction_id_list)
    instruction_correct += sum(follow_instruction_list)

    for instruction_id, followed_or_not in zip(
        instruction_id_list, follow_instruction_list
    ):
      instruction_id = instruction_id.split(":")[0]
      tier0_total[instruction_id] += 1
      if followed_or_not:
        tier0_correct[instruction_id] += 1

    for instruction_id, followed_or_not in zip(
        instruction_id_list, follow_instruction_list
    ):
      tier1_total[instruction_id] += 1
      if followed_or_not:
        tier1_correct[instruction_id] += 1
  res = {}
  with open(output_file_name_log, "a") as fw:
    fw.write(f"prompt-level: {prompt_correct / prompt_total}"+"\n")
    fw.write(f"instruction-level: {instruction_correct / instruction_total}"+"\n")
    fw.write("\n")
    res["prompt-level"] = float(prompt_correct / prompt_total)
    res["instruction-level"] = float(instruction_correct / instruction_total)

    for instruction_id in sorted(tier0_total.keys()):
      accuracy = tier0_correct[instruction_id] / tier0_total[instruction_id]
      fw.write(f"{instruction_id} {accuracy}"+"\n")
    fw.write("\n")
    for instruction_id in sorted(tier1_total.keys()):
      accuracy = tier1_correct[instruction_id] / tier1_total[instruction_id]
      fw.write(f"{instruction_id} {accuracy}"+"\n")
  return res



def recursively_check(root_dir):
    json_pathlist = []
    visit = [root_dir]
    while len(visit):
        cur_path = visit.pop()
        json_pathlist += glob(os.path.join(cur_path, "input_data_ans_*.jsonl"))
        if os.path.isdir(cur_path):
            subdirs = os.listdir(cur_path)
            for subdir in subdirs:
                visit.append(os.path.join(cur_path, subdir))
    json_pathlist = [item for item in json_pathlist if not (item.endswith("invalid.json"))]
    return json_pathlist


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  # INPUT_DATA_value = "/apdcephfs_cq8/share_2992827/shennong_5/theodoreguo/work/instruct_generate/self_play/IFEval_hacking/IFEval_final/IFEval_prompt_results.jsonl"
  # INPUT_RESPONSE_DATA_value = "/apdcephfs_cq8/share_2992827/shennong_5/theodoreguo/work/instruct_generate/self_play/IFEval_hacking/IFEval_final/IFEval_prompt_results.jsonl"

  INPUT_DATA_value = "/apdcephfs_cq8/share_2992827/shennong_5/theodoreguo/work/instruct_generate/self_play/IFEval_hacking/IFEval_final_supplement/IFEval_response_supplement.jsonl"
  INPUT_RESPONSE_DATA_value = "/apdcephfs_cq8/share_2992827/shennong_5/theodoreguo/work/instruct_generate/self_play/IFEval_hacking/IFEval_final_supplement/IFEval_response_supplement.jsonl"

  # inputs = read_prompt_list(_INPUT_DATA.value)
  # prompt_to_response_pathlist = recursively_check(_INPUT_RESPONSE_DATA.value)

  inputs = read_prompt_list(INPUT_DATA_value)
  prompt_to_response_pathlist = [INPUT_RESPONSE_DATA_value]
  save_total_path = os.path.join(RESULTS_ROOT, "IF-Eval")
  os.makedirs(save_total_path, exist_ok=True)

  print("inputs number", len(inputs))

  for prompt_to_response_path in tqdm(prompt_to_response_pathlist):
    output_file_name1 = os.path.join(
        os.path.dirname(prompt_to_response_path), "eval_results_strict.jsonl"
    )
    output_file_name2 = os.path.join(
        os.path.dirname(prompt_to_response_path), "eval_results_loose.jsonl"
    )
    if os.path.exists(output_file_name1) and os.path.exists(output_file_name2):
      continue
    print("Processing...{}".format(prompt_to_response_path))
    prompt_to_response = read_prompt_to_response_dict(prompt_to_response_path)
    # get instruction following results
    model_id = os.path.basename(os.path.dirname(prompt_to_response_path))
    res_combined = {}
    for func, output_file_name in [
        (test_instruction_following_strict, "eval_results_strict"),
        (test_instruction_following_loose, "eval_results_loose"),
    ]:
      logging.info("Generating %s...", output_file_name)
      outputs = []
      for inp in inputs:
        outputs.append(func(inp, prompt_to_response))
      follow_all_instructions = [o.follow_all_instructions for o in outputs]
      accuracy = sum(follow_all_instructions) / len(outputs)
      logging.info("Accuracy: %f", accuracy)

      if _OUTPUT_DIR.value == "":
        output_file_name = os.path.join(
            os.path.dirname(prompt_to_response_path), output_file_name + ".jsonl"
        )
      else:        
        output_file_name = os.path.join(
            _OUTPUT_DIR.value, output_file_name + ".jsonl"
        )
      output_file_name_log = output_file_name.replace(".jsonl", ".log")
      write_outputs(output_file_name, outputs)
      logging.info("Generated: %s", output_file_name)

      # Prints instruction following accuracy report.
      # print("=" * 64)
      # print(f"{output_file_name} Accuracy Scores:")
      res = print_report(outputs, output_file_name_log)
      res_flag = "strict" if "strict" in output_file_name else "loose"
      for res_name, res_metric in res.items():
        res_combined[res_name + "_" + res_flag] = res_metric
    res_combined["main_metric"] = float(np.mean([v for k,v in res_combined.items()]))
    final_res = {model_id:{"IFEval":res_combined}}
    
    save_res_combiend_json_path = os.path.join(save_total_path, "{}.jsonl".format(model_id))
    if os.path.exists(save_res_combiend_json_path):
        with open(save_res_combiend_json_path, "r") as fr:
            try:
                info = json.load(fr)
                if model_id in info:
                    new_add = info[model_id]
                    final_res_add = final_res[model_id]
                    final_res[model_id] = {**final_res_add, **new_add}
            except:
                pass

    # import pdb; pdb.set_trace();
    with open(save_res_combiend_json_path, "w") as fw:
        fw.write(json.dumps(final_res, indent=4, ensure_ascii=False))
  
  return


if __name__ == "__main__":
  app.run(main)
