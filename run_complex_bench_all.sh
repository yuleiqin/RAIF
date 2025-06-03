
# model_id="{YOUR MODEL ID}"
# model_url="{YOUR MODEL URL-OPENAI FORMAT: xxxx/v1/}"

model_id=$1
model_url=$2
batch_size=${3-16}

## Optional: Setting CoT(deepclaude) MODEL
cot_model_id=${4-"N/A"}
cot_model_url=${5-"N/A"}
## JUDGE MODEL
judge_model_id=${6-Qwen2.5-72B-INT8}
judge_model_url=${7-"{YOUR MODEL URL-OPEN AI FORMAT}/v1/"}


## IF-EVAL (optional)
echo ">>> Now evaluating IFEVAL benchmark"
cd COMPLEX_INSTRUCTIONS/instruction_following_eval
sh run.sh ${model_id} ${model_url} ${batch_size} ${cot_model_id} ${cot_model_url}

## CELLO
echo ">>> Now evaluating CELLO benchmark"
cd COMPLEX_INSTRUCTIONS/CELLO
sh eval.sh ${model_id} ${model_url} ${batch_size} ${cot_model_id} ${cot_model_url}

## ComplexBench
echo ">>> Now evaluating ComplexBench benchmark"
cd COMPLEX_INSTRUCTIONS/ComplexBench
sh eval_llm_qwen2.5.sh ${model_id} ${model_url} ${judge_model_id} ${judge_model_url} ${batch_size} ${cot_model_id} ${cot_model_url}

## CFBench
echo ">>> Now evaluating CFBench benchmark"
cd COMPLEX_INSTRUCTIONS/CFBench
sh run_infer_eval.sh ${model_id} ${model_url} ${judge_model_id} ${judge_model_url} ${batch_size} ${cot_model_id} ${cot_model_url}

## FB-Bench
echo ">>> Now evaluating FB-Bench benchmark"
cd COMPLEX_INSTRUCTIONS/FB-Bench
sh run_eval_qwen.sh ${model_id} ${model_url} ${judge_model_id} ${judge_model_url} ${batch_size} ${cot_model_id} ${cot_model_url}

## FollowBench
echo ">>> Now evaluating FollowBench benchmark"
cd COMPLEX_INSTRUCTIONS/FollowBench
echo ">>> Now evaluating FollowBench benchmark-英文"
sh eval_en.sh ${model_id} ${model_url} ${judge_model_id} ${judge_model_url} ${batch_size} ${cot_model_id} ${cot_model_url}
echo ">>> Now evaluating FollowBench benchmark-中文"
sh eval_zh.sh ${model_id} ${model_url} ${judge_model_id} ${judge_model_url} ${batch_size} ${cot_model_id} ${cot_model_url}

## InfoBench
echo ">>> Now evaluating InfoBench benchmark"
cd COMPLEX_INSTRUCTIONS/InfoBench
sh eval.sh ${model_id} ${model_url} ${judge_model_id} ${judge_model_url} ${batch_size} ${cot_model_id} ${cot_model_url}
