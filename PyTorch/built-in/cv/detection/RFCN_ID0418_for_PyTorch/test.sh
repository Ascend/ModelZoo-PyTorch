#!/usr/bin/env python

source ./env.sh
python3 test_net.py \
    --arch=rfcn \
    --dataset=pascal_voc \
    --net=res101 \
    --cfg=cfg/res101.yml \
    --checksession 1 \
    --checkepoch 20 \
    --checkpoint 2504 \
    --device=npu \
    --npu_id="npu:1" \
    --amp \
    --opt_level=O1 \
    --loss_scale=1024.0
