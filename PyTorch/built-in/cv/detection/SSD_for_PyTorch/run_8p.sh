source env_npu.sh
RANK_SIZE=8
RANK_ID_START=0
export WORLD_SIZE=${RANK_SIZE}
export RANK_SIZE=${RANK_SIZE}
RANK_ID=0
export RANK_ID=${RANK_ID}
export ASCEND_DEVICE_ID=$RANK_ID
ASCEND_DEVICE_ID=$RANK_ID
export RANK=${RANK_ID}
KERNEL_NUM=$(($(nproc)/8))
for((RANK_ID=0;RANK_ID<RANK_SIZE;RANK_ID++))
do
    export RANK=$RANK_ID
    if [ $(uname -m) = "aarch64" ]
    then
       PID_START=$((KERNEL_NUM * RANK_ID))
        PID_END=$((PID_START + KERNEL_NUM - 1))
        taskset -c $PID_START-$PID_END python3 ./tools/train.py configs/ssd/ssdlite_mobilenetv2_scratch_600e_coco_8p.py \
            --launcher pytorch \
            --seed 0 \
            --gpu-ids 0 \
            --opt-level O1 &
    else
        python3 ./tools/train.py configs/ssd/ssdlite_mobilenetv2_scratch_600e_coco_8p.py \
            --launcher pytorch \
            --seed 0 \
            --gpu-ids 0 \
            --opt-level O1 &
    fi
done
wait
python3 ./tools/test.py configs/ssd/ssdlite_mobilenetv2_scratch_600e_coco_8p.py ./work_dirs/ssdlite_mobilenetv2_scratch_600e_coco_8p/epoch_120.pth --eval=bbox
