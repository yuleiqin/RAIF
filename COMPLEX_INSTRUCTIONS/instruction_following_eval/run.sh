# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/bin/bash

model_name=$1
model_url=$2
batch_size=$3
cot_model_id=${4-"N/A"}
cot_model_url=${5-"N/A"}


cd COMPLEX_INSTRUCTIONS/instruction_following_eval

llm_output_root=$PWD/"inference/"${model_name}
mkdir -p ${llm_output_root}

echo "Inferring IF-EVAL..."

llm_output_raw_path=${llm_output_root}/${model_name}.jsonl

data_path_tobe_infered=$PWD/"data/input_data.jsonl"
bash run_complex_conv.sh ${data_path_tobe_infered} ${llm_output_raw_path} $model_name $model_url $batch_size $cot_model_id $cot_model_url
wait

echo "Evaluating IF-EVAL..."

python3 convert_json2jsonl.py $llm_output_raw_path

python3 evaluation_main.py \
  --input_data=${PWD}/data/input_data.jsonl \
  --input_response_data=$llm_output_root

