#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
ADDR=${ADDR:-127.0.0.1}
PORT=${PORT:-29500}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python3 -m torch.distributed.launch --nnodes=$NNODES --node_rank=$NODE_RANK \
 --nproc_per_node=$GPUS --master_addr=$ADDR --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}
