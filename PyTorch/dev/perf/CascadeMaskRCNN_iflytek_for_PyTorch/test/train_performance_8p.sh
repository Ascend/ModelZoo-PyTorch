#!/bin/bash
export HCCL_WHITELIST_DISABLE=1
export WORLD_SIZE=8
export MASTER_ADDR=localhost
export MASTER_PORT=29688
RANK_ID_START=0
DEVICE_ID_START=0
for((RANK_ID=$RANK_ID_START;RANK_ID<$((WORLD_SIZE+RANK_ID_START));RANK_ID++));
do
    export DEVICE_ID=$((DEVICE_ID_START+RANK_ID))
    echo "Device ID:" $DEVICE_ID
    export LOCAL_RANK=$RANK_ID
    export RANK=$RANK_ID
    python3 tools/train.py --work-dir ./ configs/cascade_rcnn/cascade_mask_rcnn_r50_fpn_1x_coco.py --launcher pytorch &
done
wait
