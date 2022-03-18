#!/bin/bash

################ introduction #################
# Welcome to the TinyBERT project
# The whole task-distill training includes two step.
# This is the first step: intermediate layer distillation
# Before bashing the shell file, please change the current dir(sharing the same dir with the folder"test")

################ parser setting ##################
# npu device id(using command npu-smi info -l to get the device id and set it as you want)
device_id='3'

################ compiling ##################
source ./test/env_npu_1p.sh
python3.7 ./main.py --teacher_model ./bert_base_uncased_ft_sst \
                 --student_model ./general_tinybert \
                 --data_dir ./glue_dir/SST-2 \
                 --task_name SST-2 \
                 --output_dir ./tmp_tinybert_performance \
                 --max_seq_length 64 \
                 --train_batch_size 32 \
                 --num_train_epochs 1 \
                 --aug_train \
                 --do_lower_case \
	               --device 'npu' \
	               --addr $(hostname -I |awk '{print $1}') \
	               --amp \
	               --device_list ${device_id} \
	               --port '12000' \
	               --opt_level 'O1' \
	               --performance \
	               --fps_acc_dir ./output