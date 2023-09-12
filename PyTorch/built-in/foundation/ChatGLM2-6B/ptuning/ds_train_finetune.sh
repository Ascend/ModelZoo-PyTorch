
LR=1e-4

MASTER_PORT=$(shuf -n 1 -i 10000-65535)

source env_npu.sh
deepspeed --num_gpus=8 --master_port $MASTER_PORT main_without_tokenizer.py \
    --deepspeed deepspeed.json \
    --do_train \
    --train_file AdvertiseGen/train.json \
    --test_file AdvertiseGen/dev.json \
    --prompt_column content \
    --response_column summary \
    --model_name_or_path ../../6Bv2_weight/ \
    --output_dir ./output/adgen-chatglm2-6b-ft-$LR \
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
    --fp16

