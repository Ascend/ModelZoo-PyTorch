#!/bin/bash

source scripts/set_npu_env.sh


python evalOnePic.py \
    --data_path /data/BSD68  \
    --resume net8p.pth  | tee eval.log




