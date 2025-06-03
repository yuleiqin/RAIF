
model_id=$1
model_url=${2-"localhost"}
batch_size=${3-16}
# model_id="qwen2.5_72b"
# model_id="Taskformer1.0.2"
cot_model_id=${4-"N/A"}
cot_model_url=${5-"N/A"}


# CUDA_VISIBLE_DEVICES=0 python code/eval.py --model_name chatglm --save_name chatglm
# CUDA_VISIBLE_DEVICES=0 python code/eval.py --model_name gpt4 --save_name gpt4

input_dir="COMPLEX_INSTRUCTIONS/CELLO/results/gpt4"
output_json_path="COMPLEX_INSTRUCTIONS/CELLO/cello_combined.jsonl"

infered_json_root="COMPLEX_INSTRUCTIONS/CELLO/results"
save_inferred_json_path=$infered_json_root/${model_id}.jsonl

sh run_complex_conv.sh $output_json_path $save_inferred_json_path $model_id $model_url $batch_size $cot_model_id $cot_model_url

infered_json_root_model=${infered_json_root}/${model_id}
mkdir -p ${infered_json_root_model}

python3 combine_and_split.py --input_path $save_inferred_json_path --output_path $infered_json_root_model --is_combine False --model_id ${model_id}

labeled_data_path=COMPLEX_INSTRUCTIONS/CELLO/data
saved_score_path=COMPLEX_INSTRUCTIONS/CELLO/scores

mkdir -p ${saved_score_path}

python code/score.py --result_path ${infered_json_root} --labeled_data_path ${labeled_data_path} --saved_score_path ${saved_score_path}

