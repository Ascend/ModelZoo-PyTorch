#!/usr/bin/env bash

lr="0.00001"
bs="1024"
filename="train_full_1p"
data_path = "data/FB15k-237"
for para in $*
do
    if [[ $para == --lr* ]]; then
        lr=`echo ${para#*=}`
    fi
    if [[ $para == --bs* ]]; then
        bs=`echo ${para#*=}`
    fi
    if [[ $para == --filename* ]]; then
        filename=`echo ${para#*=}`
    fi
    if [[ $para == --f* ]]; then
        filename=`echo ${para#*=}`
    fi
    if [[ $para == --data_path* ]]; then
        data_path=`echo ${para#*=}`
    fi
    if [[ $para == --data* ]]; then
        data_path=`echo ${para#*=}`
    fi
done
python codes/apex_run.py \
    --do_train \
    --do_test \
    --do_valid \
    --npu \
    --data_path ${data_path} \
    --model RotatE \
    -n 256 \
    -b ${bs} \
    -d 1000 \
    -g 9.0 \
    -a 1.0 \
    -adv \
    -lr ${lr} \
    --max_steps 150000 \
    --warm_up_steps 100000 \
    -save models/${filename} \
    --test_batch_size 16 \
    -de \
    --apex \
    --apex_level O1 \
    --loss_scale 128.0 \
    --world_size 1 \
    --port 23456 \
    -cpu 4 \
    --test_cuda \
    --backend hccl \