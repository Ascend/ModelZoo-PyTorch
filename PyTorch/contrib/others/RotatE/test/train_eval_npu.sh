#!/usr/bin/env bash

save="train_eval_npu"
for para in $*
do
    if [[ $para == --save* ]]; then
        save=`echo ${para#*=}`
    fi
done
python codes/apex_run.py \
    --do_test \
    --data_path data/FB15k-237 \
    --model RotatE \
    -n 256 \
    -d 1000 \
    -g 9.0 \
    -a 1.0 \
    -adv \
    -lr 0.00001 \
    --max_steps 150000 \
    --warm_up_steps 100000\
    -save models/${save} \
    -init models/${save} \
    --test_batch_size 16 \
    -de \
    --apex \
    --apex_level O1 \
    --loss_scale 128.0 \
    --prof \
    --world_size 1 \
    --port 23456 \
    -cpu 4 \
    --npu \