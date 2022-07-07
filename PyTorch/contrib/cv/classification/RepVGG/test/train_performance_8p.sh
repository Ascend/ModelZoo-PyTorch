#!/usr/bin/env python

source test/env_npu.sh

data_path=""
for para in $*
do
    if [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
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

nohup taskset -c $PID_START-$PID_END python3.7.5 -u train.py \
-a RepVGG-A0 \
--data ${data_path} \
--epochs 3 \
--workers 192 \
--batch-size=4096 \
--lr 1.6 \
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
--loss-scale-value "dynamic" > repvgg_8p_perf.log 2>&1 &
done
