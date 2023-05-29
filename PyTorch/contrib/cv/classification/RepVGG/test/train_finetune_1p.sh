#!/usr/bin/env bash
source test/env_npu.sh

data_path=""
for para in $*
do
    if [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    elif [[ $para == --batch_size* ]];then
        batch_size=`echo ${para#*=}`
    elif [[ $para == --device_id* ]];then
        device_id=`echo ${para#*=}`
    fi
done

if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
    exit 1
fi

nohup python3 -u train.py \
    --fresume "RepVGG-A0_hello_best.pth.tar" \
    --finetune 1 \
    --fclasses 1000 \
    -a RepVGG-A0 \
    --data ${data_path} \
    --workers 32 \
    --custom-weight-decay \
    --epochs 1 \
    --batch-size=${batch_size} \
    --lr 0.2 \
    --wd 4e-5 \
    --device npu \
    --rank_id ${device_id} \
    --amp \
    --opt-level "O2" \
    --loss-scale-value "dynamic" > repvgg_finetune.log 2>&1 &
