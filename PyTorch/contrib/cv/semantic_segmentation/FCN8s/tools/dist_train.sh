#!/usr/bin/env bash

CONFIG=$1
NPUS=$2
ADDR=${ADDR:-127.0.0.1}
PORT=${PORT:-30515}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
taskset -c 0-96 python3 -m torch.distributed.launch --nnodes=$NNODES --node_rank=$NODE_RANK \
--nproc_per_node=$NPUS --master_addr=$ADDR --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}
