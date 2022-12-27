#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
NNODE=${NNODE:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python3.7 -u  -m torch.distributed.launch --nnodes=${NNODE} --node_rank=${NODE_RANK} --nproc_per_node=$GPUS --master_addr=${MASTER_ADDR} --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}
