

model_id=$1
model_url=$2
judge_model_id=${3-Qwen2.5-72B-INT8}
judge_model_url=${4-https://ms-xfr6fl5w-100034032793-sw.gw.ap-shanghai.ti.tencentcs.com/ms-xfr6fl5w/v1/}
batch_size=${5-64}
cot_model_id=${6-"N/A"}
cot_model_url=${7-"N/A"}


cd COMPLEX_INSTRUCTIONS/InfoBench

# 开始推理
output_root=infer
mkdir -p $output_root
output_dir=$output_root/$model_id/
mkdir -p $output_dir

input=COMPLEX_INSTRUCTIONS/InfoBench/InFoBench/InfoBench.jsonl
output_path=$PWD/$output_dir/$model_id.jsonl
echo "Output JSON PATH: "$output_path
sh run_complex.sh $input $output_path $model_id $model_url ${batch_size} $cot_model_id $cot_model_url

wait

judge_dir=$PWD/$output_dir/judge_by_${judge_model_id}
mkdir -p $judge_dir

python evaluation.py \
  --model ${judge_model_id} \
  --model_url ${judge_model_url} \
  --input ${output_path} \
  --output_dir ${judge_dir} \
  --temperature 0 --batch_size ${batch_size} --compute_metric 0

input=$judge_dir/$judge_model_id/$model_id"_DecomposeEval.json"
output_path=$judge_dir/$judge_model_id/$model_id"_DecomposeEval_ans.jsonl"

echo "Input path: "$input
echo "Output JSON PATH: "$output_path
sh run_complex2.sh $input $output_path $judge_model_id $judge_model_url ${batch_size}

wait

python evaluation.py \
  --model ${judge_model_id} \
  --model_url ${judge_model_url} \
  --input ${output_path} \
  --output_dir ${judge_dir} \
  --temperature 0 --batch_size ${batch_size} --compute_metric 1

