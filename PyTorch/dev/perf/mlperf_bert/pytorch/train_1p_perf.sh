#!/bin/bash

data_and_model_path=./

python3 run_pretraining.py \
  --input_dir=$data_and_model_path/2048_shards_uncompressed \
  --eval_dir=$data_and_model_path/eval_set_uncompressed \
  --learning_rate=2e-5 \
  --bert_model="bert-large-uncased" \
  --output_dir='./checkpoint/' \
  --train_mlm_accuracy_window_size=0 \
  --warmup_proportion=0 \
  --warmup_steps=0 \
  --start_warmup_step=0 \
  --target_mlm_accuracy=0.720 \
  --weight_decay_rate=0.01 \
  --eval_iter_start_samples=125000 \
  --eval_iter_samples=125000 \
  --max_seq_length=512 \
  --cache_eval_data \
  --max_steps=100 \
  --max_predictions_per_seq=76 \
  --train_batch_size=10 \
  --eval_batch_size=10 \
  --max_samples_termination=14000000 \
  --gradient_accumulation_steps=1 \
  --do_train \
  --bert_config_path=$data_and_model_path/bert_config.json \
  --dense_seq_outpu \
  --init_checkpoint=$data_and_model_path/model.ckpt-28252.pt > 1p_perf.log 2>&1 & \