#!/bin/bash

################ introduction #################
# Welcome to the TinyBERT project
# The whole task-distill training includes two step.
# This is the second step: prediction layer distillation
# Before bashing the shell file, please change the current dir(sharing the same dir with the folder"test")

################ parser setting ##################
# npu device id(using command npu-smi info -l to get the device id and set it as you want)
device_id='0,1,2,3,4,5,6,7'

################ compiling ##################
source ./test/env_npu_8p.sh
python3.7 ./main.py \
	      --pred_distill  \
        --teacher_model ./bert_base_uncased_ft_sst \
        --student_model ./tmp_tinybert_dir \
        --data_dir ./glue_dir/SST-2 \
        --task_name SST-2 \
        --output_dir ./TinyBERT_dir \
        --aug_train \
        --learning_rate 3e-5 \
        --num_train_epochs  3 \
        --eval_step 100 \
        --max_seq_length 64 \
        --train_batch_size 32 \
	      --do_lower_case \
	      --workers 1 \
	      --device 'npu' \
	      --addr $(hostname -I |awk '{print $1}') \
	      --amp \
	      --device_list ${device_id} \
	      --port '12000' \
	      --loss_scale 128 \
	      --opt_level 'O2' \
	      --fps_acc_dir ./output
