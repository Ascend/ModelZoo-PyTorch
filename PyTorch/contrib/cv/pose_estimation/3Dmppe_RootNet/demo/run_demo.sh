#!/bin/bash

source test/env_npu.sh
cd demo
rm -f nohup.out

nohup python3 -u demo.py \
    --npu_device 0 \
    --test_epoch 18 &
