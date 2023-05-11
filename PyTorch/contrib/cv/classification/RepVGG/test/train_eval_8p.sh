#!/usr/bin/env python

# 训练batch_size
batch_size=4096

source test/env_npu.sh

data_path=""
for para in $*
do
    if [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    elif [[ $para == --batch_size* ]];then
        batch_size=`echo ${para#*=}`
    fi
done

if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
    exit 1
fi

RANK_ID_START=0
RANK_SIZE=8

for((RANK_ID=$RANK_ID_START;RANK_ID<$((RANK_SIZE+RANK_ID_START));RANK_ID++));
do

KERNEL_NUM=$(($(nproc)/8))
PID_START=$((KERNEL_NUM * RANK_ID))
PID_END=$((PID_START + KERNEL_NUM - 1))

nohup taskset -c $PID_START-$PID_END python3 -u train.py \
--resume "RepVGG-A0_hello_best.pth.tar" \
--evaluate \
-a RepVGG-A0 \
--data ${data_path} \
--workers 192 \
--batch-size=${batch_size} \
--lr 0.1 \
--wd 4e-5 \
--device npu \
--num_gpus 8 \
--rank_id $RANK_ID \
--addr "127.0.0.1" \
--port "29588" \
--dist-backend "hccl" \
--tag hello \
--custom-weight-decay \
--amp \
--opt-level "O2" \
--loss-scale-value "dynamic" > repvgg_eval.log 2>&1 &
done
