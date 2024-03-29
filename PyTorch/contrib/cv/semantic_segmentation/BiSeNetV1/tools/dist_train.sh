#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
#PORT=${PORT:-29511}
PORT=${PORT:-30515}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
taskset -c 0-96 python3 -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}
