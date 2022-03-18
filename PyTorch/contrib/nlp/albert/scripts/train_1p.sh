# encoding=utf-8

source ./scripts/env_npu.sh

export BERT_BASE_DIR=./prev_trained_model/albert_base_v2
export DATA_DIR=./dataset
export OUTPUR_DIR=./outputs
export DEVICE=npu
TASK_NAME="SST-2"

# batch x 20
python3.7 ./run_classifier.py \
  --device=$DEVICE \
  --model_type=$BERT_MODEL \
  --model_name_or_path=$BERT_BASE_DIR/$BERT_MODEL \
  --task_name=$TASK_NAME \
  --data_dir=$DATA_DIR/$TASK_NAME/ \
  --spm_model_file=$BERT_BASE_DIR/$BERT_MODEL/30k-clean.model \
  --output_dir=$OUTPUR_DIR/$TASK_NAME/ \
  --do_train \
  --do_eval \
  --do_lower_case \
  --max_seq_length=128 \
  --batch_size=440 \
  --learning_rate=28e-5 \
  --num_train_epochs=2.0 \
  --logging_steps=80 \
  --save_steps=80 \
  --overwrite_output_dir \
  --seed=42 \
  --fp16 \
  --fp16_opt_level=O2
