import json
import sys
import os


if __name__ == '__main__':
    model_name = sys.argv[1]
    os.makedirs('./results', exist_ok=True)
    ans = {}
    with open(f'./inference/{model_name}/eval_results_loose.log') as fin:
        for line in fin:
            line = line.strip()
            if line.startswith('prompt-level'):
                ans['prompt-level_loose'] = line.split(' ')[-1]
            elif line.startswith('instruction-level'):
                ans['instruction-level_loose'] = line.split(' ')[-1]
            else:
                break
    with open(f'./inference/{model_name}/eval_results_strict.log') as fin:
        for line in fin:
            line = line.strip()
            if line.startswith('prompt-level'):
                ans['prompt-level_strict'] = line.split(' ')[-1]
            elif line.startswith('instruction-level'):
                ans['instruction-level_strict'] = line.split(' ')[-1]
            else:
                break
    ans['main_metric'] = ans['prompt-level_loose']

    output_ans = {model_name: {"IFEval": ans}}
    json.dump(output_ans, open(f'./results/{model_name}.jsonl', 'w'), ensure_ascii=False, indent=4)