

model_id=$1
model_url=$2
judge_model_id=$3
judge_model_url=$4
batch_size=${5-16}
cot_model_id=${6-"N/A"}
cot_model_url=${7-"N/A"}


output_root=inference
mkdir -p inference
output_dir=inference/$model_id

cd COMPLEX_INSTRUCTIONS/FB-Bench
#================================================================
#===============      STEP 1                   ==================
#================================================================
#================================================================
# 起服务 推理以下文件
###############################################################
python3 gen_answer_local.py --model_id $model_id --out_dir $output_dir --model_url $model_url --batch_size ${batch_size} --cot_model_id ${cot_model_id} --cot_model_url ${cot_model_url}

# judge_model_id="qwen2.5_72b"
eval_out_dir=judgement/$model_id/judge-$judge_model_id
log_dir=$eval_out_dir/logs
mkdir -p $log_dir

python3 gen_judgment_local.py --model_id $model_id --out_dir $PWD/$output_dir \
    --judge_out_dir $PWD/$eval_out_dir --judge_model_id $judge_model_id --judge_model_url $judge_model_url \
        --batch_size ${batch_size}

wait

python3 show_results.py --judge_dir $PWD/$eval_out_dir
