#!/usr/bin/env bash
source scripts/npu_set_env.sh

nohup python3.7.5 -u main.py \
    --data /home/dataset/imagenet \
    --workers 192 \
    --epoch 100 \
    --batch-size=1024 \
    --lr 0.4 \
    --momentum 0.9 \
    --wd 1e-4 \
    --device npu \
    --device-list '0,1,2,3,4,5,6,7' \
    --multiprocessing-distributed \
    --world-size 1 \
    --rank 0 \
    --addr "127.0.0.1" \
    --port "29588" \
    --dist-backend "hccl" \
    --amp \
    --opt-level "O2" \
    --eval-freq 1 \
    --resume "./model_best.pth.tar" \
    --evaluate \
    --loss-scale-value 16 > seresnext50_32x4d_eval_8p.log 2>&1 &
