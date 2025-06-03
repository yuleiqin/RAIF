
input_path=$1
save_path=$2
model_id=$3
model_url=$4
batch_size=${5-64}
cot_model_id=${6-"N/A"}
cot_model_url=${7-"N/A"}


echo "batch_size=${batch_size}"
question_type=q
qkey=prompt
input_file_type=jsonl
save_freq=10
max_tokens=2048
top_k=1
temperature=0

# system_prompt="A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. Always answer briefly in Chinese. "
# system_prompt="A chat between a user and an artificial intelligence assistant. The assistant provides succinct, brief, direct, and concise answers to the user's questions without redundant explanation, always getting straight to the point. Always responds in the same language as the user. "
system_prompt="You are a helpful assistant."

python3 COMPLEX_INSTRUCTIONS/run_local_model_openai.py \
--input_path $input_path \
--input_file_type $input_file_type \
--save_path $save_path \
--model_id $model_id \
--model_url $model_url \
--question_type $question_type \
--qkey $qkey \
--save_freq $save_freq \
--batch_size $batch_size \
--max_tokens $max_tokens \
--top_k $top_k \
--temperature $temperature \
--cot_model_id $cot_model_id \
--cot_model_url $cot_model_url \
--resume
