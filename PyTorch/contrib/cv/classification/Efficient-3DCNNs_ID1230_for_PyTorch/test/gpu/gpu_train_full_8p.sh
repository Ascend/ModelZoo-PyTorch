#!/usr/bin/env bash
root_path=$1

nohup python3 -m torch.distributed.launch --nproc_per_node 8 ../main.py \
    --root_path ${root_path} \
    --gpu_or_npu gpu \
    --use_prof 0 \
    --use_apex 1 \
    --device_lists 0,1,2,3,4,5,6,7 \
    --distributed 1 \
    --n_classes 101 \
    --n_finetune_classes 101 \
    --learning_rate 0.08 \
    --droupout_rate 0.9 \
    --n_epochs 30 \
    --batch_size 640 \
    --n_threads 64 \
	  --ft_portion complete \
	  > ${root_path}/results/gpu_train_full_8p.log 2>&1 &
