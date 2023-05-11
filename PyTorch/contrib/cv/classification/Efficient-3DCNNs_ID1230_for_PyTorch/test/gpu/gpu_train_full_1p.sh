#!/usr/bin/env bash
root_path=$1

nohup python3 ../main.py \
    --root_path ${root_path} \
    --gpu_or_npu gpu \
    --use_prof 1 \
    --use_apex 1 \
    --device_lists 0 \
    --distributed 0 \
    --n_classes 101 \
    --n_finetune_classes 101 \
	  --learning_rate 0.01 \
	  --droupout_rate 0.9 \
	  --n_epochs 2 \
	  --batch_size 80 \
	  --n_threads 16 \
	  --ft_portion complete \
	  > ${root_path}/results/gpu_train_full_1p.log 2>&1 &



