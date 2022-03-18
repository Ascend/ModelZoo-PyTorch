#!/bin/bash

python train_roberta.py \
	-data roberta-data-bin/demo \
	-save_model roberta-demo-model \
	--batch_type tokens \
	--batch_size 4096 \
	--gpu_ranks 0 \
	--single_pass
