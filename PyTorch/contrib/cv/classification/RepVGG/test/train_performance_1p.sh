#!/usr/bin/env bash
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

nohup python3.7.5 -u train.py \
    -a RepVGG-A0 \
    --data ${data_path} \
    --workers 32 \
    --epochs 3 \
    --batch-size=512 \
    --lr 0.2 \
    --wd 4e-5 \
    --device npu \
    --amp \
    --custom-weight-decay \
    --opt-level "O2" \
    --loss-scale-value "dynamic" > repvgg_1p_perf.log 2>&1 &
