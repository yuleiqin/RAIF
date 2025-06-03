#!/bin/bash
HDFS_HOME={YOUR_ROOT_PATH}/
mkdir -p $HDFS_HOME
timestamp=`date +'%Y%m%d_%H'`0000
WANDB_KEY="{YOUR_WANDB_KEY}"

EXPERIMENT_NAME="R1_simpleRL_deepscalar_complexIF"
RUN_NAME=DS-QWEN-7B_grpo_DCI_1-1_SFT_supCoT

DATA_PATH="data/DeepScaleR-IFEval-Complex/DeepSeek-R1-Distill-Qwen-7B/数学_全量指令_ratio_1_1_combined.jsonl"
SFT_DATA_PATH="data/IFEval-Complex/DeepSeek-R1-Distill-Qwen-7B/all_data.jsonl"

PRETRAIN_PATH="{YOUR_ROOT_PATH}/models/Qwen2.5-7B-Instruct_Qwen"
PRETRAIN_PATH_SFT="{YOUR_ROOT_PATH}/models/DeepSeek-R1-Distill-Qwen-7B"
CKPT_PATH=$HDFS_HOME/checkpoints_nips/$RUN_NAME
mkdir -p ${CKPT_PATH}

IS_DEBUG="False"
TRAIN_BATCH_SIZE=128
ROLLOUT_BATCH_SIZE=32
MICRO_TRAIN_BATCH_SIZE=1
MICRO_ROLLOUT_BATCH_SIZE=1

NUM_SAMPLES_PER_PROMPT=8 ## DEFAULT
TEMPERATURE=1.0
# save train script
CURRENT_SCRIPT_PATH="$(realpath $0)"
cp "${CURRENT_SCRIPT_PATH}" "${CKPT_PATH}"

pip3 install langdetect
pip3 install nltk
pip install immutabledict

python3 openrlhf/cli/train_ppo_ray_box.py \
    --ref_num_nodes 1 \
    --ref_num_gpus_per_node 4 \
    --reward_num_nodes 1 \
    --reward_num_gpus_per_node 4 \
    --critic_num_nodes 0 \
    --critic_num_gpus_per_node 0 \
    --actor_num_nodes 1 \
    --actor_num_gpus_per_node 4 \
    --reward_pretrain $PRETRAIN_PATH \
    --critic_pretrain "None" \
    --pretrain $PRETRAIN_PATH_SFT \
    --colocate_actor_ref \
    --vllm_num_engines 8 \
    --vllm_tensor_parallel_size 1 \
    --save_path $HDFS_HOME/checkpoints/$RUN_NAME \
    --micro_train_batch_size ${MICRO_TRAIN_BATCH_SIZE} \
    --train_batch_size ${TRAIN_BATCH_SIZE} \
    --micro_rollout_batch_size ${MICRO_ROLLOUT_BATCH_SIZE} \
    --rollout_batch_size ${ROLLOUT_BATCH_SIZE} \
    --temperature ${TEMPERATURE} \
    --n_samples_per_prompt ${NUM_SAMPLES_PER_PROMPT} \
    --max_samples 100000 \
    --max_epochs 1 \
    --num_episodes 30 \
    --advantage_estimator "group_norm" \
    --enforce_superior_CoT \
    --use_kl_loss \
    --use_kl_estimator_k3 \
    --prompt_max_len 8192 \
    --generate_max_len 4096 \
    --zero_stage 3 \
    --bf16 \
    --actor_learning_rate 1e-6 \
    --init_kl_coef 0.001 \
    --gamma 1.0 \
    --prompt_data $DATA_PATH \
    --input_key input \
    --label_key target \
    --pretrain_data $SFT_DATA_PATH \
    --input_key_pretrain question \
    --label_key_pretrain answer \
    --normalize_reward \
    --flash_attn \
    --adam_offload \
    --gradient_checkpointing \
    --save_steps 50 \
    --load_checkpoint \
    --save_hf_ckpt \
    --use_wandb $WANDB_KEY \
    --wandb_run_name $RUN_NAME \
    --wandb_project $EXPERIMENT_NAME \
    --ckpt_path ${CKPT_PATH}  \
    --max_ckpt_num 2 \
    --ptx_coef 1
