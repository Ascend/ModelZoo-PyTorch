#!/usr/bin/env bash
source ./test/env_npu.sh

python3 train_kinetics.py \
	--batch-size 4 \
	--gpus 0 \
	--dataset ucf101 \
	--apex yes \
	--apex_level O2 \
	--loss_scale 128.0 \
	--lr-base 0.001 \
	--prof no \
	--end-epoch 3