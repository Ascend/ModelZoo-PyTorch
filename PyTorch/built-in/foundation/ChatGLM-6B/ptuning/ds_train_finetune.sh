LR=1e-4

MASTER_PORT=$(shuf -n 1 -i 10000-65535)
export TRAIN_STATE=1
source env_npu.sh
deepspeed --num_gpus=8 --master_port $MASTER_PORT main.py \
    --deepspeed deepspeed.json \
    --do_train \
    --train_file AdvertiseGen/train.json \
    --test_file AdvertiseGen/dev.json \
    --prompt_column content \
    --response_column summary \
    --overwrite_cache \
    --enable_process \
    --model_name_or_path ../model/ \
    --output_dir ./output/adgen-chatglm-6b-ft-$LR-test \
    --overwrite_output_dir \
    --max_source_length 1024 \
    --max_target_length 1024 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --predict_with_generate \
    --max_steps 5000 \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate $LR \
    --fp16 \
    --dataloader_drop_last \
    --ddp_timeout 7200

