#!/usr/bin/env bash
source ./test/env_npu.sh

python3 test.py \
        -c=configs/m2det512_vgg.py \
        -m=weights/M2Det_COCO_size512_netvgg16_epoch150.pth \
        --device_list=0,1,2,3,4,5,6,7 \
        --device='npu'
