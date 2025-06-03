

model_id=$1
input_jsonl_path=$2
judge_model_id=$3
eval_out_dir=$4
judge_model_url=$5
eval_max_threads=${6-16}
mkdir -p $eval_out_dir
eval_score_path=$eval_out_dir/$model_id-scores.xlsx


python  code/evalaute_local.py \
    --infer_model ${model_id} \
    --in_dir ${input_jsonl_path} \
    --out_dir ${eval_out_dir}  \
    --score_path ${eval_score_path} \
    --max_threads ${eval_max_threads} \
    --eval_model ${judge_model_id} \
    --eval_model_url $judge_model_url

