CURRENT_DIR=`pwd`/classifier_pytorch
export BERT_BASE_DIR=$CURRENT_DIR/prev_trained_model/bert-base-chinese
export GLUE_DIR=$CURRENT_DIR/chineseGLUEdatasets
export OUTPUR_DIR=$CURRENT_DIR/outputs
TASK_NAME="tnews"

export PYTHONPATH=./classifier_pytorch:$PYTHONPATH

export MASTER_ADDR='127.0.0.1'
export MASTER_PORT='23333'
export WORLD_SIZE=4

RANK_ID_START=0
KERNEL_NUM=$(($(nproc) / 8))

for ((RANK_ID = $RANK_ID_START; RANK_ID < $((WORLD_SIZE + RANK_ID_START)); RANK_ID++))
do
	PID_START=$((KERNEL_NUM * $RANK_ID))
	PID_END=$((PID_START + KERNEL_NUM - 1))
	export RANK=$RANK_ID
	taskset -c $PID_START-$PID_END nohup python -u \
	run.py \
	--local_rank $RANK_ID \
	--fp16 \
	--model_type=bert \
	--model_name_or_path=$BERT_BASE_DIR \
	--task_name=$TASK_NAME \
	--do_train \
	--do_eval \
	--do_lower_case \
	--data_dir=$GLUE_DIR/${TASK_NAME}/ \
	--max_seq_length=128 \
	--per_gpu_train_batch_size=256 \
	--per_gpu_eval_batch_size=32 \
	--learning_rate=6e-5 \
	--num_train_epochs=4.0 \
	--logging_steps=8372 \
	--save_steps=8372 \
	--output_dir=$OUTPUR_DIR/${TASK_NAME}_output/ \
	--overwrite_output_dir \
	&
done

