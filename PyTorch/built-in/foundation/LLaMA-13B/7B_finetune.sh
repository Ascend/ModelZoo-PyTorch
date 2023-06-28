source /usr/local/Ascend/ascend-toolkit/set_env.sh

export HCCL_CONNECT_TIMEOUT=1200

deepspeed --num_gpus=8  \
    fastchat/train/train_mem.py \
    --model_name_or_path ./7B-vicuna/ \
    --data_path ./playground/data/alpaca-data-conversation.json \
    --fp16 True \
    --output_dir ./ckpt_8p \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 200 \
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
    --deepspeed ./deepspeed_config_7B.json | tee train_7B.log