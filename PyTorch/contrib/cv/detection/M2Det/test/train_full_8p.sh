#!/usr/bin/env bash
source ./test/env_npu.sh

taskset -c 0-95 python3 -u train_8p.py \
        --c=configs/m2det512_vgg.py \
        --ngpu=8 \
        --amp \
        --opt_level=O1 \
        --loss_scale=32.0 \
        --world_size=1 \
        --dist-url='tcp://127.0.0.1:30003' \
        --dist_backend=hccl \
        --dist_rank=0 \
        --multiprocessing_distributed \
        --device_list=0,1,2,3,4,5,6,7 \
        --device='npu' \
        --addr=$(hostname -I |awk '{print $1}') \
        --seed=49 \
        --FusedSGD
        
        
