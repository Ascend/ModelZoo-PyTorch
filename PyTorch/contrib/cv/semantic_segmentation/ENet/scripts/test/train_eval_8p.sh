#!/bin/bash

source env_npu.sh

python3 proc_node_module.py

nohup python3 -m torch.distributed.launch --nproc_per_node=8 eval.py \
        --model enet \
        --dataset citys \
        > train_eval_8p.log &