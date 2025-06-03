

## 待评测模型的model_id
model_id=$1
## 待评测模型的model_url
model_url=$2

## 待打分的模型的judge_model_id
judge_model_id=$3
## 待打分的模型的judge_model_url
judge_model_url=$4
batch_size=${5-16}
cot_model_id=${6-"N/A"}
cot_model_url=${7-"N/A"}


input_jsonl_path=data/cfbench_data_infer.jsonl
output_jsonl_path=inference/$model_id.jsonl

cd COMPLEX_INSTRUCTIONS/CFBench
#================================================================
#===============      STEP 1                   ==================
#================================================================
#================================================================
# 起服务 推理以下文件
###############################################################
sh run_infer.sh $PWD/$input_jsonl_path $PWD/$output_jsonl_path $model_id $model_url ${batch_size} ${cot_model_id} ${cot_model_url}

convert_jsonl_dir=evaluation/$model_id
mkdir -p evaluation/$model_id
convert_jsonl_path=$convert_jsonl_dir/$model_id.json

python3 process/convert_jsonl.py --input_path $output_jsonl_path --output_path $convert_jsonl_path --model_id $model_id
wait

# judge_model_id="qwen2.5_72b"
eval_out_dir=judgement/$model_id/judge_$judge_model_id
log_dir=$eval_out_dir/logs
mkdir -p $log_dir

sh run_eval.sh $model_id $convert_jsonl_path $judge_model_id $eval_out_dir $judge_model_url ${batch_size}

wait
