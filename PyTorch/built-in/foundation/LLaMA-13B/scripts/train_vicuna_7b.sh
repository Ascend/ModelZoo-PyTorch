source /home/flj/CANN/ascend-toolkit/set_env.sh
DATE_TIME=`date +'%Y_%m_%d_%H_%M_%S'`
mkdir ./../logs
path=./../logs/train_7B_$DATE_TIME.log


torchrun --nproc_per_node=8 --master_port=20001 fastchat/train/train_mem.py \
    --model_name_or_path /home/flj/llama2-7b-hf  \
    --data_path /home/flj/FastChat-1106/data/alpaca.json \
    --bf16 True \
    --output_dir ckpt \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "steps" \
    --eval_steps 1500 \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.04 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True | tee  $path 2>&1 &

