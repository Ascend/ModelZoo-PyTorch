#!/bin/bash

################ introduction #################
# Welcome to the TinyBERT project
# This is the finetune shell(transfer learning)
# It means the program can load the previous model files to continue training
# and the classifier can adjust to your dataset automatically(the number of output nodes of the nn.Linear changes)
# So, if you want to do the transfer learning in this project, all you need to do is to just change your dataset.

################ parser setting ##################
# npu device id(using command npu-smi info -l to get the device id and set it as you want)
device_id='3'

################ compiling ##################
source ./test/env_npu_1p.sh
python3 ./main.py --teacher_model ./bert_base_uncased_ft_mnli \
                 --student_model ./general_tinybert \
                 --data_dir ./glue_dir/MNLI \
                 --task_name mnli \
                 --output_dir ./tmp_tinybert_finetune \
                 --max_seq_length 64 \
                 --train_batch_size 32 \
                 --num_train_epochs 10 \
                 --do_lower_case \
	               --device 'npu' \
	               --addr $(hostname -I |awk '{print $1}') \
	               --amp \
	               --device_list ${device_id} \
	               --port '12000' \
	               --opt_level 'O1' \
	               --fps_acc_dir ./output \
	               --transfer_learning