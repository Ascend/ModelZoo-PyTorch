#!/bin/bash

################ introduction #################
# Welcome to the TinyBERT project
# This is the final step: evaluation!

################ parser setting ##################
# npu device id(using command npu-smi info -l to get the device id and set it as you want)
device_id='3'

################ compiling ##################
source ./test/env_npu_1p.sh
python3 main.py \
  		 --do_eval \
		   --student_model ./TinyBERT_dir \
		   --data_dir ./glue_dir/SST-2 \
			 --task_name SST-2 \
			 --output_dir ./result_dir \
			 --do_lower_case \
			 --max_seq_length 64 \
			 --eval_batch_size 32 \
			 --device 'npu' \
	     --addr $(hostname -I |awk '{print $1}') \
	     --amp \
	     --device_list ${device_id} \
	     --port '12000' \
	     --fps_acc_dir ./output