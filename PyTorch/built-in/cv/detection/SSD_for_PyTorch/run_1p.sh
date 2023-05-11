source env_npu.sh
RANK_SIZE=1
RANK_ID_START=0
export WORLD_SIZE=${RANK_SIZE}
export RANK_SIZE=${RANK_SIZE}
RANK_ID=0
export RANK_ID=${RANK_ID}
export ASCEND_DEVICE_ID=$RANK_ID
ASCEND_DEVICE_ID=$RANK_ID
export RANK=${RANK_ID}
python3 ./tools/train.py configs/ssd/ssdlite_mobilenetv2_scratch_600e_coco_1p.py \
    --seed 0 \
    --gpu-ids 0 \
    --opt-level O1
wait
