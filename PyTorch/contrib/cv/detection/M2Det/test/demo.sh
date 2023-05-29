#!/usr/bin/env bash
source ./test/env_npu.sh

python3 -u demo.py \
        -c=configs/m2det512_vgg.py \
        -m=weights/m2det512_vgg.pth \
        --device='npu:0'
