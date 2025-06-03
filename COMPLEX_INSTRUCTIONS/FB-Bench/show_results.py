import os
import sys
import json
import pandas as pd
import random
import csv
import ast
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba, to_hex
from matplotlib import cm
import seaborn as sns
# from adjustText import adjust_text
import numpy as np
from collections import Counter, defaultdict
import random
import json
import yaml
from glob import glob
from pathlib import Path
from scipy.stats import spearmanr
import pprint
import math
import copy
from math import pi
import argparse


defined_task_types = ['Mathematics', "Reasoning", "Coding", "Text Extraction", "Text Error Correction", "Text Creation", "Knowledge Q&A", "Text Translation"]
defined_error_types = ["Not Following Instructions", "Logical Error", "Incomplete Answer", "Factual Error", "Unprofessional Answer"]
defined_feedback_types_correct = ["Pointing Out Errors", "Simple Questioning", "Clarifying Intent", "Raising Objections", "Detailed Explanation", "Hinting Guidance"]
defined_feedback_types_antisyco = ["Misinformation", "Simple Questioning", "Credibility Support", "Unreasonable Requests"]

def get_avg_score_of_nest_list(data, float_num=4):
    tmp = []
    for item in data:
        if isinstance(item, list):
            tmp.append( sum(item)/len(item) )
        else:
            tmp.append(item)
    res =  round(sum(tmp)/len(tmp), float_num)

    return res

def get_avg_score(data, float_num=4, mode='type'):
    correct_data = []
    antisyco_data = []

    correct_data_per_task = {task: [] for task in defined_task_types}
    antisyco_data_per_task = {task: [] for task in defined_task_types}
    
    for item in data:
        if item['bench_type'] == "Error Correction":
            correct_data.append(item['score'])
            correct_data_per_task[item['task_type']].append(item['score'])  # {'Math': [], 'Coding': [], ...}
        elif item['bench_type'] == "Response Maintenance":
            antisyco_data.append(item['score'])
            antisyco_data_per_task[item['task_type']].append(item['score'])
        else:
            raise ValueError(f"Unknown bench_type: {item['bench_type']}, ") 
    all_data = correct_data + antisyco_data
    
    if mode == 'item':
        correct_avg_score = round(sum(correct_data)/len(correct_data), float_num) if correct_data else None
        antisyco_avg_score = round(sum(antisyco_data)/len(antisyco_data), float_num) if antisyco_data else None
        all_avg_score = round(sum(all_data)/len(all_data), float_num) if all_data else None
    elif mode == 'type':
        correct_data_per_task = list(correct_data_per_task.values())  # [ [], [], ..., [] ]
        antisyco_data_per_task = list(antisyco_data_per_task.values())
        all_data_per_task = correct_data_per_task + antisyco_data_per_task
        correct_data_per_task = [item for item in correct_data_per_task if item]  # [ [x, y, ..., z], [], ..., [] ] ——> [[x, y, ..., z]]
        antisyco_data_per_task = [item for item in antisyco_data_per_task if item]
        all_data_per_task = [item for item in all_data_per_task if item]
        
        correct_avg_score = get_avg_score_of_nest_list(correct_data_per_task, float_num) if correct_data_per_task else None
        antisyco_avg_score = get_avg_score_of_nest_list(antisyco_data_per_task, float_num) if antisyco_data_per_task else None
        all_avg_score = get_avg_score_of_nest_list(all_data_per_task, float_num) if all_data_per_task else None
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    correct_avg_score = correct_avg_score * 100 if correct_avg_score else None
    antisyco_avg_score = antisyco_avg_score * 100 if antisyco_avg_score else None
    all_avg_score = all_avg_score * 100 if all_avg_score else None

    
    return correct_avg_score, antisyco_avg_score, all_avg_score



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--judge_dir", type=str, default="data/feedback-benchmark/model_judgment/gpt-4o-2024-08-06")
    args = parser.parse_args()
    judge_dir_path = args.judge_dir
    assert(os.path.exists(judge_dir_path))
    # judge_dir = Path("data/feedback-benchmark/model_judgment/gpt-4o-2024-08-06")
    judge_dir = Path(judge_dir_path)
    filenames = glob(os.path.join(judge_dir, "*.jsonl"))
    overall_score = []
    for filename in filenames:
        model_name = os.path.splitext(os.path.basename(filename))[0]
        with open(filename, 'r') as f:
            data = [json.loads(l.strip()) for l in f]
            overall_score.append((model_name, *get_avg_score(data)))

    df_overall_score = pd.DataFrame(overall_score, columns=["model_name", "error_correction_score", "response_maintenance_score", "overall_score"]).sort_values(by="overall_score", ascending=False)

    df_overall_score.reset_index(drop=True, inplace=True)
    df_overall_score.index = range(1, len(df_overall_score) + 1)

    print(df_overall_score)
    csv_path = os.path.join(judge_dir, "results.csv")
    df_overall_score.to_csv(csv_path)