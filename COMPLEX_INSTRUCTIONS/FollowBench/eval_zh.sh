

model_id=$1
model_url=$2
judge_model_id=$3
judge_model_url=$4
batch_size=${5-16}
cot_model_id=${6-"N/A"}
cot_model_url=${7-"N/A"}


cd COMPLEX_INSTRUCTIONS/FollowBench
#### Model Inference
### convert to Chinese inference version
# python code_zh/model_inference.py

# 开始推理
input_root_dir=data_infer_zh
output_root=inference_zh
mkdir -p $output_root
output_dir=$output_root/$model_id/raw
mkdir -p $output_dir

for input in `realpath $input_root_dir/*`
do
    echo "Input JSON PATH: "$input
    file_name=$(basename "$input")
    output_path=$PWD/$output_dir/$file_name
    echo "Output JSON PATH: "$output_path
    sh run_complex.sh $input $output_path $model_id $model_url ${batch_size} $cot_model_id $cot_model_url
    wait
done

output_dir_processed=$output_root/$model_id/processed
mkdir -p $output_dir_processed
python3 convert_raw2processed.py --input_path_local $output_dir --output_path $output_dir_processed --model_id $model_id


#### LLM-based Evaluation
# 开始judge打分
evaluation_root=evaluation_zh
mkdir -p $evaluation_root
gpt4_discriminative_eval_input_path=$evaluation_root/$model_id/gpt4_discriminative_eval_input
data_gpt4_discriminative_eval_input_path=$evaluation_root/$model_id/data_gpt4_discriminative_eval_input
mkdir -p $gpt4_discriminative_eval_input_path
mkdir -p $data_gpt4_discriminative_eval_input_path

# judge_model_id="qwen2.5_72B"

log_dir=$evaluation_root/$model_id/judged_by_$judge_model_id/log
mkdir -p $log_dir
evaluation_dir=$evaluation_root/$model_id/judged_by_$judge_model_id/tobeinfered
mkdir -p $evaluation_dir
python code_zh/llm_eval.py --api_output_path $output_dir_processed --gpt4_discriminative_eval_output_path $evaluation_dir \
    --model_path $model_id --gpt4_discriminative_eval_input_path $gpt4_discriminative_eval_input_path \
        --data_gpt4_discriminative_eval_input_path $data_gpt4_discriminative_eval_input_path

evaluation_res=$evaluation_root/$model_id/judged_by_$judge_model_id/infered
mkdir -p $evaluation_res
evaluation_res_processed=$evaluation_root/$model_id/judged_by_$judge_model_id/infered_processed
mkdir -p $evaluation_res_processed
evaluation_result_path=$evaluation_root/$model_id/judged_by_$judge_model_id/evaluation_final
mkdir -p $evaluation_result_path
for input in `realpath $evaluation_dir/*`
do
    echo "Input JSON PATH: "$input
    file_name=$(basename "$input")
    output_path=$PWD/$evaluation_res/$file_name
    echo "Output JSON PATH: "$output_path
    sh run_complex.sh $input $output_path $judge_model_id $judge_model_url ${batch_size}
    wait
done

wait

python3 convert_raw2processed.py --input_path_local $evaluation_res --output_path $evaluation_res_processed --judge_model_id $judge_model_id
# python3 convert_raw2processed.py --input_path_local $evaluation_res2 --output_path $evaluation_res_processed2 --judge_model_id $judge_model_id2


#### Merge Evaluation and Save Results 
# Next, we conduct **rule-based evaluation** and merge the **rule-based evaluation** results and **LLM-based evaluation** results using the following script:

python3 code_zh/eval.py --model_paths $model_id --api_output_path $output_dir_processed --gpt4_discriminative_eval_output_path $evaluation_res_processed \
    --data_gpt4_discriminative_eval_input_path $data_gpt4_discriminative_eval_input_path \
        --evaluation_result_path $evaluation_result_path

# python3 code_zh/eval.py --model_paths $model_id --api_output_path $output_dir_processed --gpt4_discriminative_eval_output_path $evaluation_res_processed2 \
#     --data_gpt4_discriminative_eval_input_path $data_gpt4_discriminative_eval_input_path \
#         --evaluation_result_path $evaluation_result_path2
