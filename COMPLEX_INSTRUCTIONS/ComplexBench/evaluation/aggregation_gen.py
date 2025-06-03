import json
import argparse
import os
from copy import deepcopy

def load_jsonl(file_path):
    _data = []
    with open(file_path, 'r') as f:
        for data in f:
            jline = json.loads(data)
            _data.append(jline)
    return _data


def get_rely_judgment(entry):
    point_judges_rely = deepcopy(entry['point_judges'])
    for i, q in enumerate(entry['scoring_questions']):
        for dep in q['dep']:
            if entry['point_judges'][dep] == False:
                point_judges_rely[i] = False
    entry['point_judges_rely'] = point_judges_rely
    return entry


def get_selection_depth(entry):
    depth = 0
    for q in entry['scoring_questions']:
        if 'Selection' in q['composition_types']:
            depth += 1
    
    return depth


def get_nested_method(entry):
    if "_".join(sorted(entry['composition_types'])) == "And_And_Selection":
        print(entry)
    return "_".join(sorted(entry['composition_types']))


Lexical = ["Word Matching", "Keywords"]
Structure = ["JSON Format", "Markdown Format", "Bullets Format", "Length", "Start with", "End with", "Punctuation", "Template"]
Semantic = ["Language Style", "Personalization", "Topic", "Sentiment"]
Utillity = ["Helpfulness", "Target Language", "Supportiveness", "Consistency", "Factuality"]

Tasks = ["Fundamental Language Ability", "Advanced Chinese Understanding", "Open-ended Questions", "Practical Writing", "Creative Writing", "Professional Writing", "Custom Writing", "Logical Reasoning", "Task-oriented Role Play", "Professional Knowledge"]


def get_type(text):
    if text in Lexical:
        return "Lexical"
    if text in Structure:
        return "Structure"
    if text in Semantic:
        return "Semantic"
    if text in Utillity:
        return "Utillity"
    return "Others"
        

def aggregation(_data):
    categories_cnt = {}
    categories_acc = {}

    categories_avg_cnt = {}
    categories_avg_acc = {}

    constraint_types_and_composition_types_cnt = {}
    constraint_types_and_composition_types_acc = {}

    each_cnt = {}
    each_acc = {}
    
    task_types_cnt = {}
    task_types_acc = {}

    nested_methods_cnt = {}
    nested_methods_acc = {}

    count_true = 0
    count_false = 0

    for entry in _data:
        for i, p in enumerate(entry['scoring_questions']):
            for constraint in p['constraint_dimensions']:
                type = get_type(constraint)
                constraint_types_and_composition_types_cnt[type] = constraint_types_and_composition_types_cnt.get(type, 0) + 1
                each_cnt[constraint] = each_cnt.get(constraint, 0) + 1 
                if entry['point_judges_rely'][i]:
                    constraint_types_and_composition_types_acc[type] = constraint_types_and_composition_types_acc.get(type, 0) + 1
                    each_acc[constraint] = each_acc.get(constraint, 0) + 1
                
            for composition in p['composition_types']:
                constraint_types_and_composition_types_cnt[composition] = constraint_types_and_composition_types_cnt.get(composition, 0) + 1
                each_cnt[composition] = each_cnt.get(composition, 0) + 1
                if entry['point_judges_rely'][i]:
                    constraint_types_and_composition_types_acc[composition] = constraint_types_and_composition_types_acc.get(composition, 0) + 1
                    each_acc[composition] = each_acc.get(composition, 0) + 1
                
        category_avg = entry['category'].split("_")[0]
        nested_method = get_nested_method(entry)
        categories_cnt[entry['category']] = categories_cnt.get(entry['category'], 0) + len(entry['scoring_questions'])
        categories_avg_cnt[category_avg] = categories_avg_cnt.get(category_avg, 0) + len(entry['scoring_questions'])
        task_types_cnt[entry['task_types']] = task_types_cnt.get(entry['task_types'], 0) + len(entry['scoring_questions'])
        nested_methods_cnt[nested_method] = nested_methods_cnt.get(nested_method, 0) + len(entry['scoring_questions'])

        for i in entry['point_judges_rely']:
            if i == True:
                categories_acc[entry['category']] = categories_acc.get(entry['category'], 0) + 1
                categories_avg_acc[category_avg] = categories_avg_acc.get(category_avg, 0) + 1
                task_types_acc[entry['task_types']] = task_types_acc.get(entry['task_types'], 0) + 1
                nested_methods_acc[nested_method] = nested_methods_acc.get(nested_method, 0) + 1
                count_true += 1
            else:
                count_false += 1
        
    for t in constraint_types_and_composition_types_cnt.keys():
        constraint_types_and_composition_types_acc[t] = constraint_types_and_composition_types_acc.get(t, 0) / constraint_types_and_composition_types_cnt[t]
    
    for t in categories_cnt.keys():
        categories_acc[t] = categories_acc.get(t, 0) / categories_cnt[t]
    
    for t in categories_avg_cnt.keys():
        categories_avg_acc[t] = categories_avg_acc.get(t, 0) / categories_avg_cnt[t]
    
    for t in nested_methods_cnt.keys():
        nested_methods_acc[t] = nested_methods_acc.get(t, 0) / nested_methods_cnt[t]
    
    categories_acc = {k: categories_acc[k] for k in sorted(categories_acc)}
    categories_avg_acc = {k: categories_avg_acc[k] for k in sorted(categories_avg_acc)}
    nested_methods_acc = {k: nested_methods_acc[k] for k in sorted(nested_methods_acc)}

    for t in each_cnt.keys():
        each_acc[t] = each_acc.get(t, 0) / each_cnt[t]
    
    for t in task_types_cnt.keys():
        task_types_acc[t] = task_types_acc.get(t, 0) / task_types_cnt[t]

    # Coherent Test
    data_by_id = {}
    tree_1 = []
    tree_2 = []

    for entry in _data:
        data_by_id[entry['group']] = data_by_id.get(entry['group'], [])
        data_by_id[entry['group']].append(entry)
    
    cnt = 0
    for k in data_by_id.keys():
        depth = get_selection_depth(data_by_id[k][0])
        if depth == 1:
            tree_1.append(data_by_id[k])
        elif depth >= 2:
            tree_2.append(data_by_id[k])
            

    tree_1_acc = 0
    tree_2_acc = 0

    total_1_acc = 0
    total_2_acc = 0

    total_1_cnt = 0
    total_2_cnt = 0

    for t in tree_1:
        flag = True
        for sample in t:
            total_1_cnt += 1
            fg = True
            for q in sample['point_judges_rely']:
                if q == False:
                    flag = False
                    fg = False
            
            if fg == True:
                total_1_acc += 1
        if flag == True:
            tree_1_acc += 1


    for t in tree_2:
        flag = True
        for sample in t:
            total_2_cnt += 1
            fg = True
            for q in sample['point_judges_rely']:
                if q == False:
                    flag = False
                    fg = False
            
            if fg == True:
                total_2_acc += 1
        if flag == True:
            tree_2_acc += 1
    
    return {
        "overall_drfr" : count_true / (count_true + count_false),
        "categorized_drfr" : categories_acc,
        "categorized_avg_drfr" : categories_avg_acc,
        "task_types_drfr" : task_types_acc,
        "nested_methods_drfr" : nested_methods_acc,
        "constraint_types_and_composition_types_acc" : constraint_types_and_composition_types_acc,
        "each_constraint_dimensions_and_composition_types_acc" : each_acc,
        "single_origin_test" : total_1_acc / (total_1_cnt+1e-6),
        "single_coherent_test" : tree_1_acc / (len(tree_1)+1e-6),
        "single_relative_drop" : (total_1_acc / (total_1_cnt+1e-6) - tree_1_acc / ((len(tree_1))+1e-6) / ((total_1_acc / (total_1_cnt+1e-6))+1e-6)+1e-6),
        "multiple_origin_test" : total_2_acc / (total_2_cnt+1e-6),
        "multiple_coherent_test" : tree_2_acc / (len(tree_2)+1e-6),
        "miltiple_relative_drop" : (total_2_acc / (total_2_cnt+1e-6) - tree_2_acc / ((len(tree_2)+1e-6)) / ((total_2_acc / (total_2_cnt+1e-6))+1e-6)+1e-6),
    }





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--llm_evaluation_path", type=str, default="")
    parser.add_argument("--rule_evaluation_path", type=str, default="")
    parser.add_argument("--output_path", type=str, default="")
    parser.add_argument("--model", type=str, default="")
    args = parser.parse_args()
    model = args.model

    print("args.data_path", args.data_path)
    with open(args.data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print("len(data)=", len(data))
    outputs = {}
    point_judges = {}
    point_explanation = {}
    
    ## 存储llm evaluation的评价结果
    llm_evaluations = load_jsonl(args.llm_evaluation_path)
    for l in llm_evaluations:
        ## 模型输出的结果
        outputs[l['main_id']] = l['output']
        ## point_id对应的是多个限制下的得分点编号
        ## point_judge对应的是这个得分点是否满足了
        # point_explanation对应的是这个得分点评价的理由
        point_judges[(l['main_id'], l['point_id'])] = l['point_judge']
        point_explanation[(l['main_id'], l['point_id'])] = l['point_explanation']
    
    rule_evaluations = load_jsonl(args.rule_evaluation_path)
    for r in rule_evaluations:
        ## rule-based直接根据运算符就能得到true/false的取值
        if r['point_judge'] == 0:
            r['point_judge'] = False
        if r['point_judge'] == 1:
            r['point_judge'] = True
        ## 模型输出的结果
        outputs[l['main_id']] = l['output']
        ## point_judge对应的是这个得分点是否满足了
        point_judges[(r['main_id'], r['point_id'])] = r['point_judge']
        ## point_explanation对应的是这个得分点评价的理由
        point_explanation[(r['main_id'], r['point_id'])] = r['point_explanation']
    
    outs = []
    outs_invalid = []
    for da in data:
        ### main_id是唯一标志符
        ### 所有样本呢的输出都唯一根据这个标志符来进行指向
        # if not (da['main_id'] in outputs):
        #     outs_invalid.append(da)
        #     continue
        da['output'] = outputs[da['main_id']]
        da['point_judges'] = []
        da['point_explanations'] = []
        is_invalid = False
        for i in range(len(da['scoring_questions'])):
            if not ((da['main_id'], i) in point_judges):
                import pdb;pdb.set_trace();
                is_invalid = True
                continue
                # break
            da['point_judges'].append(point_judges[(da['main_id'], i)])
            da['point_explanations'].append(point_explanation[(da['main_id'], i)])
        
        if is_invalid:
            outs_invalid.append(da)
            continue
        da = get_rely_judgment(da)
        outs.append(da)
    
    with open(os.path.join(args.output_path, f"{model}_final_results.json"), "w", encoding='utf-8') as f:
        json.dump(outs, f, ensure_ascii=False, indent=4)

    if len(outs_invalid):
        with open(os.path.join(args.output_path, f"{model}_final_results_invalid.json"), "w", encoding='utf-8') as f:
            json.dump(outs_invalid, f, ensure_ascii=False, indent=4)
    
    results = {model : aggregation(outs)}

    with open(os.path.join(args.output_path, f"{model}_statistics.json"), "w", encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

