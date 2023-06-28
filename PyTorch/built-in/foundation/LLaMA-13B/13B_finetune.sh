#! /bin/bash

# Change for multinode config
NUM_WORKERS=2
NUM_GPUS_PER_WORKER=8
MASTER_PORT=12345

DATESTR=$(date +"%m-%d-%H-%M")

HOST_FILE_PATH="./hostfile"

source /usr/local/Ascend/ascend-toolkit/set_env.sh

run_cmd="HCCL_CONNECT_TIMEOUT=1200 deepspeed --master_port ${MASTER_PORT} --num_nodes ${NUM_WORKERS} --num_gpus ${NUM_GPUS_PER_WORKER} --hostfile ${HOST_FILE_PATH} fastchat/train/train_mem.py \
    --model_name_or_path ./13B-vicuna \
    --data_path ./playground/data/alpaca-data-conversation.json \
    --fp16 True \
    --output_dir ./ckpt_16p \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length 1024 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --deepspeed ./deepspeed_config_13B.json > train_13B.log 2>&1 &"


echo ${run_cmd}
eval ${run_cmd}

set +x
