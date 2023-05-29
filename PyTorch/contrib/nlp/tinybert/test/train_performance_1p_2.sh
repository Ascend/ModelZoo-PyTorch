#!/bin/bash

################ introduction #################
# Welcome to the TinyBERT project
# The whole task-distill training includes two step.
# This is the second step: prediction layer distillation
# Before bashing the shell file, please change the current dir(sharing the same dir with the folder"test")

################ parser setting ##################
# npu device id(using command npu-smi info -l to get the device id and set it as you want)
device_id='3'

################ compiling ##################
source ./test/env_npu_1p.sh
python3 ./main.py \
	      --pred_distill  \
        --teacher_model ./bert_base_uncased_ft_sst \
        --student_model ./tmp_tinybert_performance \
        --data_dir ./glue_dir/SST-2 \
        --task_name SST-2 \
        --output_dir ./TinyBERT_dir_performance \
        --aug_train \
        --learning_rate 3e-5 \
        --num_train_epochs 1 \
        --eval_step 100 \
        --max_seq_length 64 \
        --train_batch_size 32 \
	      --do_lower_case \
	      --workers 1 \
	      --device 'npu' \
	      --addr $(hostname -I |awk '{print $1}') \
	      --amp \
	      --device_list '3' \
	      --port '1200' \
	      --loss_scale 128 \
	      --opt_level 'O2' \
	      --performance \
	      --fps_acc_dir ./output
