#!/usr/bin/env bash
PYTHON=${PYTHON:-"python3.7.5"}

CONFIG=configs/cityscapes_fast_scnn.yaml
GPUS=8
$PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS ./tools/train.py \
        --config-file $CONFIG ${@:3}\
        --world_size=8 \
