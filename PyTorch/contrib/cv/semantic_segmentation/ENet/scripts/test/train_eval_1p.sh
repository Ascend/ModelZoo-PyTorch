#!/bin/bash

source env_npu.sh

python3 proc_node_module.py

nohup python3 eval.py \
        --model enet \
        --dataset citys \
        > train_eval_1p.log &