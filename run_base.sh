export http_proxy="http://172.31.255.10:8888"
export https_proxy="http://172.31.255.10:8888"
export no_proxy=".woa.com,mirrors.cloud.tencent.com,tlinux-mirror.tencent-cloud.com,tlinux-mirrorlist.tencent-cloud.com,localhost,127.0.0.1,mirrors-tlinux.tencentyun.com,.oa.com,.local,.3gqq.com,.7700.org,.ad.com,.ada_sixjoy.com,.addev.com,.app.local,.apps.local,.aurora.com,.autotest123.com,.bocaiwawa.com,.boss.com,.cdc.com,.cdn.com,.cds.com,.cf.com,.cjgc.local,.cm.com,.code.com,.datamine.com,.dvas.com,.dyndns.tv,.ecc.com,.expochart.cn,.expovideo.cn,.fms.com,.great.com,.hadoop.sec,.heme.com,.home.com,.hotbar.com,.ibg.com,.ied.com,.ieg.local,.ierd.com,.imd.com,.imoss.com,.isd.com,.isoso.com,.itil.com,.kao5.com,.kf.com,.kitty.com,.lpptp.com,.m.com,.matrix.cloud,.matrix.net,.mickey.com,.mig.local,.mqq.com,.oiweb.com,.okbuy.isddev.com,.oss.com,.otaworld.com,.paipaioa.com,.qqbrowser.local,.qqinternal.com,.qqwork.com,.rtpre.com,.sc.oa.com,.sec.com,.server.com,.service.com,.sjkxinternal.com,.sllwrnm5.cn,.sng.local,.soc.com,.t.km,.tcna.com,.teg.local,.tencentvoip.com,.tenpayoa.com,.test.air.tenpay.com,.tr.com,.tr_autotest123.com,.vpn.com,.wb.local,.webdev.com,.webdev2.com,.wizard.com,.wqq.com,.wsd.com,.sng.com,.music.lan,.mnet2.com,.tencentb2.com,.tmeoa.com,.pcg.com,www.wip3.adobe.com,www-mm.wip3.adobe.com,mirrors.tencent.com,csighub.tencentyun.com,mirrors.tencentyun.com,mirrors.tencent.com"

# export NCCL_SOCKET_IFNAME=eth0
# export NCCL_IB_DISABLE=0
# export NCCL_IB_GID_INDEX=3
# export NCCL_P2P_DISABLE=0
# export NCCL_IB_CUDA_SUPPORT=1

echo $PWD

# echo "INSTALL THE PACKAGE"
# pip3 install -v -e .
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
MASTER_PORT=6379

echo "MASTER_ADDR=${MASTER_ADDR}, MASTER_PORT=${MASTER_PORT}"

# launch the master node of ray in container
echo "Now, running on node index $NODE_RANK"
# 设置内部字段分隔符为逗号
IFS=','

# 将字符串分割成数组
read -ra NODE_SUBLIST <<< "${NODE_LIST}"

NODE_SUBADDR=${NODE_SUBLIST[${NODE_RANK}]}
NODE_SUBADDR_IP="${NODE_SUBADDR%:*}"
echo "CURRENT IP ADDRESS=${NODE_SUBADDR_IP}"

## 查询每个具体的IP地址
# apt update
# apt install dnsutils -y
# NODE_SUBADDR_IP_REAL=$(nslookup $DOMAIN | awk '/^Address: / { print $2 }' | tail -n1)
# echo "IP Address: $NODE_SUBADDR_IP_REAL"

if [ "${NODE_RANK}" != "0" ]; then
    # if you want to launch ray on more nodes, use
    echo "Start NODE RANK $NODE_RANK"
    # ray start --address ${MASTER_ADDR}:${MASTER_PORT}  --num-gpus 8
    ray start --address ${MASTER_ADDR}:${MASTER_PORT}  --num-gpus 8 --node-ip-address=${NODE_SUBADDR_IP}

else
    echo "Start MASTER NODE RANK $NODE_RANK"
    # ray start --head --node-ip-address=0.0.0.0 --num-gpus=8 --port=${MASTER_PORT}
    ray start --head --node-ip-address=${MASTER_ADDR} --num-gpus=8 --port=${MASTER_PORT}
fi

# pip3 install wandb
SUBMIT_MASTER_PORT=8265

if [ "$NNODES" = "1" ]; then
    echo "Start single-node ray submit"
    ray job submit --address="http://127.0.0.1:$SUBMIT_MASTER_PORT" \
            --runtime-env-json='{
            "pip": ["ray==2.12.0", "latex2sympy2", "timeout_decorator"]
        }' -- /bin/bash examples/script/train_ppo_qwen_base_math_lv35_1_base_node.sh

else
    echo "Start multi-node ray submit"
    if [ "${NODE_RANK}" = "0" ]; then
        echo "only submit multi-node training from the master"
        ray job submit --address="http://127.0.0.1:$SUBMIT_MASTER_PORT" \
                --runtime-env-json='{
                "pip": ["ray==2.12.0", "latex2sympy2", "timeout_decorator"]
            }' -- /bin/bash examples/script/train_ppo_qwen_base_math_lv35_new.sh
    else
        echo "other nodes waiting"
    fi

fi

sleep 365d
