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

nohup taskset -c $PID_START-$PID_END python3.7.5 -u train/main.py \
--datadir ${data_path} \
--decoder \
--pretrainedEncoder "trained_models/erfnet_encoder_pretrained.pth.tar" \
--amp \
--opt-level "O2" \
--loss-scale-value 128 \
--device npu \
--num_gpus 8 \
--num-workers 32 \
--batch-size 24 \
--lr 8e-4 \
--num-epochs 10 \
--rank_id $RANK_ID > erfnet_8p_perf.log 2>&1 &
done
