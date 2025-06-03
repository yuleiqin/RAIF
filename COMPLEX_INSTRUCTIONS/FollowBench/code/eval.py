import argparse
import os
from rule_based_evaluation import save_evaluate_example_constraint, save_csl_example_constraint
from gpt4_based_evaluation import save_discriminative_evaluation, save_csl_evaluation
import numpy as np
from copy import deepcopy
import json



def main(args):

    ### rule-based evaluation
    save_evaluate_example_constraint(
                                    data_path=args.data_path, 
                                    api_output_path=args.api_output_path, 
                                    model_names=args.model_paths,
                                    evaluation_result_path=args.evaluation_result_path
        )
    
    save_csl_example_constraint(
                                data_path=args.data_path, 
                                api_output_path=args.api_output_path,
                                model_names=args.model_paths,
                                evaluation_result_path=args.evaluation_result_path
                                )


    ### LLM-based evaluation
    csl_results_by_models = {}
    hsr_results_by_models_level = {}
    hsr_results_by_models_constraints = {}
    ssr_results_by_models_level = {}
    ssr_results_by_models_constraints = {}

    for constraint_type in args.constraint_types:
        results_hsr, results_ssr = save_discriminative_evaluation(
                                        data_path=args.data_path,
                                        api_output_path=args.api_output_path,
                                        data_gpt4_discriminative_eval_input_path=args.data_gpt4_discriminative_eval_input_path, 
                                        gpt4_discriminative_eval_output_path=args.gpt4_discriminative_eval_output_path, 
                                        constraint_type=constraint_type, 
                                        model_names=args.model_paths,
                                        evaluation_result_path=args.evaluation_result_path
                                    )
        
        for hsr_pair in results_hsr:
            model_name, level1, level2, level3, level4, level5 = hsr_pair

            level1 = float(level1[:-1])
            level2 = float(level2[:-1])
            level3 = float(level3[:-1])
            level4 = float(level4[:-1])
            level5 = float(level5[:-1])

            if not (model_name in hsr_results_by_models_level):
                hsr_results_by_models_level[model_name] = {}
                hsr_results_by_models_level[model_name]["level1"] = {}
                hsr_results_by_models_level[model_name]["level2"] = {}
                hsr_results_by_models_level[model_name]["level3"] = {}
                hsr_results_by_models_level[model_name]["level4"] = {}
                hsr_results_by_models_level[model_name]["level5"] = {}
            
            if not (model_name in hsr_results_by_models_constraints):
                hsr_results_by_models_constraints[model_name] = {}

            hsr_results_by_models_constraints[model_name][constraint_type] = {}
            hsr_results_by_models_constraints[model_name][constraint_type]["level1"] = level1
            hsr_results_by_models_constraints[model_name][constraint_type]["level2"] = level2
            hsr_results_by_models_constraints[model_name][constraint_type]["level3"] = level2
            hsr_results_by_models_constraints[model_name][constraint_type]["level4"] = level4
            hsr_results_by_models_constraints[model_name][constraint_type]["level5"] = level5
            # import pdb;pdb.set_trace();
            hsr_results_by_models_constraints[model_name][constraint_type]["avg"] = np.mean([level1, level2, level3, level4, level5])
            # import pdb;pdb.set_trace();
            hsr_results_by_models_level[model_name]["level1"][constraint_type] = level1
            hsr_results_by_models_level[model_name]["level2"][constraint_type] = level2
            hsr_results_by_models_level[model_name]["level3"][constraint_type] = level3
            hsr_results_by_models_level[model_name]["level4"][constraint_type] = level4
            hsr_results_by_models_level[model_name]["level5"][constraint_type] = level5

        for ssr_pair in results_ssr:
            model_name, level1, level2, level3, level4, level5 = ssr_pair
            level1 = float(level1[:-1])
            level2 = float(level2[:-1])
            level3 = float(level3[:-1])
            level4 = float(level4[:-1])
            level5 = float(level5[:-1])

            if not (model_name in ssr_results_by_models_level):
                ssr_results_by_models_level[model_name] = {}
                ssr_results_by_models_level[model_name]["level1"] = {}
                ssr_results_by_models_level[model_name]["level2"] = {}
                ssr_results_by_models_level[model_name]["level3"] = {}
                ssr_results_by_models_level[model_name]["level4"] = {}
                ssr_results_by_models_level[model_name]["level5"] = {}
            
            if not (model_name in ssr_results_by_models_constraints):
                ssr_results_by_models_constraints[model_name] = {}

            ssr_results_by_models_constraints[model_name][constraint_type] = {}
            ssr_results_by_models_constraints[model_name][constraint_type]["level1"] = level1
            ssr_results_by_models_constraints[model_name][constraint_type]["level2"] = level2
            ssr_results_by_models_constraints[model_name][constraint_type]["level3"] = level3
            ssr_results_by_models_constraints[model_name][constraint_type]["level4"] = level4
            ssr_results_by_models_constraints[model_name][constraint_type]["level5"] = level5
            # import pdb;pdb.set_trace();
            ssr_results_by_models_constraints[model_name][constraint_type]["avg"] = np.mean([level1, level2, level3, level4, level5])
            
            ssr_results_by_models_level[model_name]["level1"][constraint_type] = level1
            ssr_results_by_models_level[model_name]["level2"][constraint_type] = level2
            ssr_results_by_models_level[model_name]["level3"][constraint_type] = level3
            ssr_results_by_models_level[model_name]["level4"][constraint_type] = level4
            ssr_results_by_models_level[model_name]["level5"][constraint_type] = level5

        csl_results = save_csl_evaluation(
                            data_path=args.data_path,
                            api_output_path=args.api_output_path,
                            data_gpt4_discriminative_eval_input_path=args.data_gpt4_discriminative_eval_input_path, 
                            gpt4_discriminative_eval_output_path=args.gpt4_discriminative_eval_output_path, 
                            constraint_type=constraint_type, 
                            model_names=args.model_paths,
                            evaluation_result_path=args.evaluation_result_path
                            )
        for csl_pair in csl_results:
            model_name, csl = csl_pair
            if not (model_name in csl_results_by_models):
                csl_results_by_models[model_name] = {}
            csl_results_by_models[model_name][constraint_type] = csl
    

    csl_results_by_models_cp = deepcopy(csl_results_by_models)
    for model_name in csl_results_by_models:
        res = csl_results_by_models[model_name]
        v_avg = []
        for constraint in res:
            v_avg.append(float(res[constraint]))
        
        # import pdb;pdb.set_trace();
        v_avg_float = np.mean(np.array(v_avg))
        csl_results_by_models_cp[model_name]["csl_avg"] = v_avg_float 

    hsr_results_by_models_level_cp = deepcopy(hsr_results_by_models_level)
    for model_name in hsr_results_by_models_level:
        res = hsr_results_by_models_level[model_name]
        for level in res:
            v_avg = []
            res_level = res[level]
            for k,v in res_level.items():
                v_avg.append(float(v))
            v_avg_float = float(np.mean(np.array(v_avg)))
            hsr_results_by_models_level_cp[model_name][level+"_avg"] = v_avg_float

    ssr_results_by_models_level_cp = deepcopy(ssr_results_by_models_level)
    for model_name in ssr_results_by_models_level:
        res = ssr_results_by_models_level[model_name]
        for level in res:
            v_avg = []
            res_level = res[level]
            for k,v in res_level.items():
                v_avg.append(float(v))
            v_avg_float = float(np.mean(np.array(v_avg)))
            ssr_results_by_models_level_cp[model_name][level+"_avg"] = v_avg_float

    hsr_results_by_models_constraints_cp = deepcopy(hsr_results_by_models_constraints)
    for model_name in hsr_results_by_models_constraints:
        res = hsr_results_by_models_constraints[model_name]
        for constraint in res:
            v_avg = []
            res_constraint = res[constraint]
            for k,v in res_constraint.items():
                v_avg.append(float(v))
            v_avg_float = float(np.mean(np.array(v_avg)))
            hsr_results_by_models_constraints_cp[model_name][constraint+"_avg"] = v_avg_float

    ssr_results_by_models_constraints_cp = deepcopy(ssr_results_by_models_constraints)
    for model_name in ssr_results_by_models_constraints:
        res = ssr_results_by_models_constraints[model_name]
        for constraint in res:
            v_avg = []
            res_constraint = res[constraint]
            for k,v in res_constraint.items():
                v_avg.append(float(v))
            v_avg_float = float(np.mean(np.array(v_avg)))
            ssr_results_by_models_constraints_cp[model_name][constraint+"_avg"] = v_avg_float

    print(f"\nEvaluation finished!\nThe evaluation results have been saved in '{args.evaluation_result_path}'.")
    
    with open(os.path.join(args.evaluation_result_path, "csl_results_by_models.json"), mode='w') as f:
        json.dump(csl_results_by_models_cp, f, ensure_ascii=False, indent=4)
    with open(os.path.join(args.evaluation_result_path, "hsr_results_by_models_level.json"), mode='w') as f:
        json.dump(hsr_results_by_models_level_cp, f, ensure_ascii=False, indent=4)
    with open(os.path.join(args.evaluation_result_path, "hsr_results_by_models_constraints.json"), mode='w') as f:
        json.dump(hsr_results_by_models_constraints_cp, f, ensure_ascii=False, indent=4)
    with open(os.path.join(args.evaluation_result_path, "ssr_results_by_models_level.json"), mode='w') as f:
        json.dump(ssr_results_by_models_level_cp, f, ensure_ascii=False, indent=4)
    with open(os.path.join(args.evaluation_result_path, "ssr_results_by_models_constraints.json"), mode='w') as f:
        json.dump(ssr_results_by_models_constraints_cp, f, ensure_ascii=False, indent=4)        

    return




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_paths", nargs='+', type=str, required=True, help="Paths or names of the models to be evaluated.")
    parser.add_argument("--constraint_types", nargs='+', type=str, default=['content', 'situation', 'style', 'format', 'mixed'])
    parser.add_argument("--data_path", type=str, default="data")
    parser.add_argument("--api_output_path", type=str, default="api_output")
    parser.add_argument("--data_gpt4_discriminative_eval_input_path", type=str, default="data_gpt4_discriminative_eval_input")
    parser.add_argument("--gpt4_discriminative_eval_output_path", type=str, default="gpt4_discriminative_eval_output")
    parser.add_argument("--evaluation_result_path", type=str, default="evaluation_result")

    args = parser.parse_args()

    if not os.path.exists(args.evaluation_result_path):
        os.makedirs(args.evaluation_result_path)

    main(args)
