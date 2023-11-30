#!/bin/bash

export HCCL_CONNECT_TIMEOUT=7200
export INF_NAN_MODE_ENABLE=1

NUM_NODES=2
NUM_GPUS_PER_NODE=8
MASTER_PORT=6667

HOST_FILE_PATH="./hostfile"

HCCL_CONNECT_TIMEOUT=1200 deepspeed --master_port ${MASTER_PORT} --num_nodes ${NUM_NODES} --num_gpus ${NUM_GPUS_PER_NODE} --hostfile ${HOST_FILE_PATH} src/train_bash.py \
    --stage sft \
    --model_name_or_path ./model_weight \
    --deepspeed ./ds_config_zero2.json \
    --do_train \
    --dataset alpaca_gpt4_en \
    --template default \
    --finetuning_type full \
    --output_dir ./output_sft \
    --overwrite_cache \
    --overwrite_output_dir \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --lr_scheduler_type cosine \
    --gradient_checkpointing True \
    --logging_steps 1 \
    --save_steps 100000 \
    --learning_rate 1e-6 \
    --num_train_epochs 10.0 \
    --plot_loss \
    --fp16 | tee logs/train_2.log
