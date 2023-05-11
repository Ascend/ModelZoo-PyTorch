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

nohup python3 -u train/main.py \
    --datadir ${data_path} \
    --finetune \
    --fnum 20 \
    --decoder \
    --pretrainedEncoder "trained_models/erfnet_encoder_pretrained.pth.tar" \
    --pretrainedDecoder "save/erfnet_training1/model_best.pth.tar" \
    --num-epochs 1 \
    --amp \
    --opt-level "O2" \
    --loss-scale-value "dynamic" > erfnet_finetune.log 2>&1 &
