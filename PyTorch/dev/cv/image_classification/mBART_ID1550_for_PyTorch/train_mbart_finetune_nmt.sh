#!/bin/bash

python train.py \
	-data data-bin/demo \
	-save_model mbartdemo-model \
	--batch_type tokens \
	--batch_size 4096 \
	--gpu_ranks 0 \
	--single_pass