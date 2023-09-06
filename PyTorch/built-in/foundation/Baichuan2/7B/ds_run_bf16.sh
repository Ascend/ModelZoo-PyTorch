source /usr/local/Ascend/ascend-toolkit/set_env.sh

export HCCL_CONNECT_TIMEOUT=1200

deepspeed --num_gpus=8 train_npu.py \
  --model_name_or_path Baichuan2-7B \
  --train_data data/train.jsonl \
  --eval_data data/eval.jsonl \
  --bf16 True \
  --output_dir outputs \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --gradient_checkpointing True \
  --save_strategy epoch \
  --learning_rate 1e-5 \
  --weight_decay 0. \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --logging_steps 1 \
  --tf32 False \
  --max_seq_length 1024 \
  --deepspeed ./ds_config_bf16.json | tee train_baichuan_7B.log
