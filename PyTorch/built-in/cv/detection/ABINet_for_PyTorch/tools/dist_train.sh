#!/usr/bin/env bash

if [ $# -lt 3 ]
then
    echo "Usage: bash $0 CONFIG WORK_DIR GPUS"
    exit
fi

CONFIG=$1
WORK_DIR=$2
GPUS=$3
LOAD_FROM=$4
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \

if [ ${GPUS} == 1 ]; then
    python $(dirname "$0")/train.py  $CONFIG --work-dir=${WORK_DIR} --load-from=${LOAD_FROM} ${@:5}
else
    python -m torch.distributed.launch \
        --nnodes=$NNODES \
        --master_addr=$MASTER_ADDR \
        --nproc_per_node=$GPUS \
        --master_port=$PORT \
        $(dirname "$0")/train.py \
        $CONFIG \
        --seed 0 \
        --load-from=${LOAD_FROM} \
        --work-dir=${WORK_DIR} \
        --launcher pytorch ${@:5}
fi
