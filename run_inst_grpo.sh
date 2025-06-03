#!/bin/bash


TRAIN_SCRIPT=${1-"examples/grpo_deepscalar_ifeval_complex_R1/grpo_qwen1-5B_1_1_complex_ifeval_sft_sup.sh"}
MASTER_PORT=${2-6379}

echo "TRAIN_SCRIPT=${TRAIN_SCRIPT}"

# export NCCL_SOCKET_IFNAME=eth0
# export NCCL_IB_DISABLE=0
# export NCCL_IB_GID_INDEX=3
# export NCCL_P2P_DISABLE=0
# export NCCL_IB_CUDA_SUPPORT=1

echo $PWD

# echo "INSTALL THE PACKAGE"
# pip3 install vllm==0.7.0
# pip3 install -v -e .
# pip3 install --upgrade nvidia-nccl-cu12
NODE_LIST=${NODE_LIST}
GPUS_PER_NODE=${GPU_NUM_PER_NODE}
NNODES=${NODE_NUM}
NODE_RANK=${INDEX}

if [ "$GPUS_PER_NODE" = "" ]; then
    GPUS_PER_NODE=8
fi

if [ "$NNODES" = "" ]; then
    NNODES=1
fi

if [ "$NODE_RANK" = "" ]; then
    NODE_RANK=0
fi

echo "GPUS_PER_NODE=${GPUS_PER_NODE}, NNODES=${NNODES}, NODE_RANK=${NODE_RANK}"

MASTER_ADDR=${MASTER_ADDR}
if [ "${MASTER_ADDR}" = "" ]; then
    MASTER_ADDR="127.0.0.1"
fi

echo "MASTER_ADDR=${MASTER_ADDR}, MASTER_PORT=${MASTER_PORT}"

# launch the master node of ray in container
echo "Now, running on node index $NODE_RANK"
# 设置内部字段分隔符为逗号
IFS=','

# 将字符串分割成数组
if [ "${NNODES}" = 1 ]; then
    NODE_SUBADDR_IP="127.0.0.1"
    echo "CURRENT IP ADDRESS=${NODE_SUBADDR_IP}"

else
    read -ra NODE_SUBLIST <<< "${NODE_LIST}"
    NODE_SUBADDR=${NODE_SUBLIST[${NODE_RANK}]}
    NODE_SUBADDR_IP="${NODE_SUBADDR%:*}"
    echo "CURRENT IP ADDRESS=${NODE_SUBADDR_IP}"

fi

## 查询每个具体的IP地址
# apt update
# apt install dnsutils -y
# NODE_SUBADDR_IP_REAL=$(nslookup $DOMAIN | awk '/^Address: / { print $2 }' | tail -n1)
# echo "IP Address: $NODE_SUBADDR_IP_REAL"

if [ "${NODE_RANK}" != "0" ]; then
    # if you want to launch ray on more nodes, use
    echo "Start NODE RANK $NODE_RANK"
    # ray start --address ${MASTER_ADDR}:${MASTER_PORT}  --num-gpus 8
    ray start --address=${MASTER_ADDR}:${MASTER_PORT}  --num-gpus=${GPUS_PER_NODE} --node-ip-address=${NODE_SUBADDR_IP}

else
    echo "Start MASTER NODE RANK $NODE_RANK"
    # ray start --head --node-ip-address=0.0.0.0 --num-gpus=8 --port=${MASTER_PORT}
    ray start --head --node-ip-address=${MASTER_ADDR} --num-gpus=${GPUS_PER_NODE} --port=${MASTER_PORT}
fi

# pip3 install wandb
SUBMIT_MASTER_PORT=8265

if [ "$NNODES" = "1" ]; then
    echo "Start single-node ray submit"
    ray job submit --address="http://127.0.0.1:$SUBMIT_MASTER_PORT" \
            --runtime-env-json='{
            "pip": ["ray==2.12.0", "latex2sympy2", "timeout_decorator"]
        }' -- /bin/bash ${TRAIN_SCRIPT}
    sleep 365d

else
    echo "Start multi-node ray submit"
    if [ "${NODE_RANK}" = "0" ]; then
        echo "only submit multi-node training from the master"
        ray job submit --address="http://127.0.0.1:$SUBMIT_MASTER_PORT" \
                --runtime-env-json='{
                "pip": ["ray==2.12.0", "latex2sympy2", "timeout_decorator"]
            }' -- /bin/bash ${TRAIN_SCRIPT}
    else
        echo "other nodes waiting"
        echo "START GPUS LOADING"

    fi

fi

sleep 365d
