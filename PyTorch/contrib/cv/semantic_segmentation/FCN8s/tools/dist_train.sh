#!/usr/bin/env bash

CONFIG=$1
NPUS=$2
PORT=${PORT:-30515}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
taskset -c 0-96 python3.7.5 -m torch.distributed.launch --nproc_per_node=$NPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}
