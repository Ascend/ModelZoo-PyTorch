source ./test/env_npu.sh
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_LOG_LEVEL=3
export PTCOPY_ENABLE=1
export TASK_QUEUE_ENABLE=1
export DYNAMIC_OP="ADD#MUL"
export COMBINED_ENABLE=1
export DYNAMIC_COMPILE_ENABLE=0
export EXPERIMENTAL_DYNAMIC_PARTITION=0
export ASCEND_GLOBAL_EVENT_ENABLE=0
export HCCL_WHITELIST_DISABLE=1

export RANK_SIZE=8
KERNEL_NUM=$(($(nproc)/8))

for((RANK_ID=0;RANK_ID<RANK_SIZE;RANK_ID++))
do
    export RANK=$RANK_ID

    if [ $(uname -m) = "aarch64" ]
    then
        PID_START=$((KERNEL_NUM * RANK_ID))
        PID_END=$((PID_START + KERNEL_NUM - 1))
        taskset -c $PID_START-$PID_END python3.7 ./tools/train.py configs/yolo/yolov3_d53_320_273e_coco.py \
            --launcher pytorch \
            --cfg-options \
            optimizer.lr=0.0032 \
            --seed 0 \
            --local_rank 0 &
    else
        python3.7 ./tools/train.py configs/yolo/yolov3_d53_320_273e_coco.py \
            --launcher pytorch \
            --cfg-options \
            optimizer.lr=0.0032 \
            --seed 0 \
            --local_rank 0 &
    fi
done
