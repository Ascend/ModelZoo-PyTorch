#!/usr/bin/env bash

lr="0.00001"
bs="1024"
filename="train_perf_8p"
index=0
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
    if [[ $para == --index* ]]; then
        index=`echo ${para#*=}`
    fi
    if [[ $para == --i* ]]; then
        index=`echo ${para#*=}`
    fi
done
python -m torch.distributed.launch --nproc_per_node 8 codes/apex_run.py \
    --do_train \
    --prof \
    --npu \
    --data_path data/FB15k-237 \
    --model RotatE \
    -n 256 \
    -b ${bs} \
    -d 1000 \
    -g 9.0 \
    -a 1.0 \
    -adv \
    -lr ${lr} \
    --max_steps 1000 \
    --warm_up_steps 150000 \
    -save models/${filename} \
    --test_batch_size 16 \
    -de \
    --apex \
    --apex_level O0 \
    --loss_scale 1.0 \
    --distributed \
    --world_size 8 \
    --port 29688 \
    -cpu 8 \
    --test_cuda \
    --backend hccl \
