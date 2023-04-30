CURRENT_DIR=`pwd`
export BERT_BASE_DIR=$CURRENT_DIR/prev_trained_model/ernie-base
export GLUE_DIR=$CURRENT_DIR/chineseGLUEdatasets
export OUTPUR_DIR=$CURRENT_DIR/outputs
TASK_NAME="tnews"
python run_classifier.py \
  --model_type=bert \
  --model_name_or_path=$BERT_BASE_DIR \
  --task_name=$TASK_NAME \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir=$GLUE_DIR/${TASK_NAME}/ \
  --max_seq_length=128 \
  --per_gpu_train_batch_size=32 \
  --per_gpu_eval_batch_size=32 \
  --learning_rate=2e-5 \
  --num_train_epochs=4.0 \
  --logging_steps=8372 \
  --save_steps=8372 \
  --output_dir=$OUTPUR_DIR/${TASK_NAME}_output/ \
  --overwrite_output_dir

# 每一个epoch保存一次
# 每一个epoch评估一次