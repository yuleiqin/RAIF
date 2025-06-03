
### 待推理模型的模型名称
model_name=$1
### 待推理模型的模型地址
model_url=$2

### judge model的模型名称
model_id=${3:-"qwen2.5_72B"}
### judge model的模型地址
model_id_url=${4:-"localhost"}
batch_size=${5:-16}
cot_model_id=${6-"N/A"}
cot_model_url=${7-"N/A"}


echo "待评价的模型为"${model_name}
echo "使用的打分模型为"${model_id}

data_path="data/data_release.json"
api_key="PLACEHOLDER"
api_base="PLACEHOLDER"
output_dir="evaluation_results"/$model_name

llm_output_raw_path=llm_generations_vllm/${model_name}_raw.jsonl
llm_output_path=llm_generations_vllm_processed/${model_name}_raw.jsonl
extraction_tobeinfered_path=${output_dir}/${model_name}_llm_extraction_results_tobeinfered.jsonl
extraction_infered_path=${output_dir}/${model_id}/${model_name}_llm_extraction_results_${model_id}.jsonl
llm_evaluation_tobeinfered_path=${output_dir}/${model_name}_llm_evaluation_results_tobeinfered.jsonl
evaluation_infered_path=${output_dir}/${model_id}/${model_name}_llm_evaluation_results_${model_id}.jsonl
extraction_path=${output_dir}/${model_id}/${model_name}_llm_extraction_results_by_${model_id}.jsonl
rule_path=${output_dir}/${model_id}/${model_name}_rule_evaluation_results_${model_id}.jsonl
evaluation_path=${output_dir}/${model_id}/${model_name}_llm_evaluation_results_by_${model_id}.jsonl
output_final_path=$output_dir/${model_id}

mkdir -p $output_dir
mkdir -p $output_dir/$model_id

#================================================================
#===============      STEP 1                   ==================
#================================================================
#================================================================
# 起服务 推理以下文件
###############################################################
data_path_tobe_infered="data_infer/data_release.jsonl"
bash run_complex.sh $PWD/${data_path_tobe_infered} $PWD/${llm_output_raw_path} $model_name $model_url ${batch_size} ${cot_model_id} ${cot_model_url}
wait
###############################################################
# 等待推理${model_name}，并将结果存放在{llm_output_raw_path}中
###############################################################
python3 data_infer/preprocess.py --input_path_local $llm_output_raw_path --save_path $llm_output_path

### 使用LLM来从答案中提取基于rule规则判断的部分语句
### 将每个问题的内容拆解出来
python3 evaluation/llm_based_extraction.py \
    --data_path $data_path \
    --llm_output_path $llm_output_path \
    --output_path ${extraction_tobeinfered_path} \
    --api_key $api_key \
    --api_base $api_base
###############################################################
# 等待推理{model_id}，并将结果存放在extraction_infered_path中
###############################################################
###  使用LLM来判断的部分语句
python3 evaluation/llm_based_evaluation.py \
    --data_path $data_path \
    --llm_output_path $llm_output_path \
    --output_path ${llm_evaluation_tobeinfered_path} \
    --api_key $api_key \
    --api_base $api_base
###############################################################
# 等待推理{model_id}，并将结果存放在evaluation_infered_path中
###############################################################


#================================================================
#===============      开始漫长等待推理              ================
#================================================================
#================================================================
bash run_complex.sh $PWD/${extraction_tobeinfered_path} $PWD/${extraction_infered_path} $model_id $model_id_url ${batch_size}
wait
bash run_complex.sh $PWD/${llm_evaluation_tobeinfered_path} $PWD/${evaluation_infered_path} $model_id $model_id_url ${batch_size}
wait
#================================================================
#===============      STEP 2a                   =================
#================================================================
#================================================================
##  将打分结果（批量推理）转换为要求的格式
python3 evaluation/extract_answers.py --input_path_local $extraction_infered_path --output_path $extraction_path --model_id $model_id

### 指标计算这些规则的满足情况
python3 evaluation/rule_based_evaluation.py \
    --data_path $data_path \
    --extraction_path ${extraction_path} \
    --output_path ${rule_path}


#================================================================
#===============      STEP 2b                   =================
#================================================================
#================================================================
##  将打分结果（批量推理）转换为要求的格式
python3 evaluation/judge_answers.py --input_path_local $evaluation_infered_path --output_path $evaluation_path --model_id $model_id

#================================================================
#===============      STEP 3                   =================
#================================================================
#================================================================
###  汇总格式 & 计算指标
python3 evaluation/aggregation.py \
    --data_path $data_path \
    --llm_evaluation_path ${evaluation_path} \
    --rule_evaluation_path ${rule_path} \
    --model $model_name \
    --output_path $output_final_path

