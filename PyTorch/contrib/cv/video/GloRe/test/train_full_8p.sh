#!/usr/bin/env bash
source ./test/env_npu.sh

python3 -m torch.distributed.launch --nproc_per_node=8 train_kinetics.py \
	--batch-size 16 \
	--gpus 0,1,2,3,4,5,6,7 \
	--dataset ucf101 \
	--world-size 8 \
	--dist-url env:// \
	--distributed yes \
	--apex yes \
	--apex_level O2 \
	--loss_scale 128.0 \
	--lr-base 0.004 \
	--end-epoch 90