#!/usr/bin/env bash
root_path=$1

nohup python3 ../main.py \
    --root_path ${root_path} \
    --gpu_or_npu gpu \
    --use_prof 1 \
    --use_apex 1 \
    --device_lists 0 \
    --distributed 0 \
	  --n_classes 600 \
	  --n_finetune_classes 101 \
	  --learning_rate 0.01 \
	  --n_epochs 2 \
	  --batch_size 80 \
	  --n_threads 16 \
	  --pretrain_path pretrain/kinetics_mobilenetv2_1.0x_RGB_16_best.pth \
	  --ft_portion last_layer \
	  > ${root_path}/results/gpu_train_full_1p_withgroup.log 2>&1 &




