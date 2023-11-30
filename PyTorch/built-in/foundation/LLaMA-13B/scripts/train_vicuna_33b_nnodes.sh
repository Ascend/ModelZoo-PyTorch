set -ex

DATESTR=$(date +"%m-%d-%H-%M-%S")
if [ ! -d ./nnodes_log ]; then
    mkdir nnodes_log
fi

source /usr/local/Ascend/ascend-toolkit/set_env.sh

torchrun \
--nproc_per_node=8 \
--nnodes=4 \
--node_rank=0 \
--master_addr=127.0.0.1  \
--master_port=29500 \
fastchat/train/train_mem.py \
    --model_name_or_path ~/checkpoints/vicuna-33b-v1.3  \
    --data_path ~/dataset/alpaca.json \
    --bf16 True \
    --output_dir output_33B \
    --num_train_epochs 3 \
    --per_device_train_batch_size 12 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "steps" \
    --eval_steps 1500 \
    --save_strategy "steps" \
    --save_steps 1500 \
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
    --lazy_preprocess True \
    --report_to tensorboard \
    --log_level info | tee ./nnodes_log/train_33b_nnodes_${DATESTR}.log 2>&1 &
