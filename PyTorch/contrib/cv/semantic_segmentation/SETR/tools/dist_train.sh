#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}


# try non dist

# 将configs/_base_中的参数设置 SyncBN修改为BN
# 改的SETR_PUP_768x768_40k_cityscapes_bs_8.py文件

