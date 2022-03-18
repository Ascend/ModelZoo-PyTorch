rm -rf kernel_meta/
/usr/local/Ascend/driver/tools/msnpureport -d 0 -g error
/usr/local/Ascend/driver/tools/msnpureport -d 4 -g error

export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_AICPU_PATH=/usr/local/Ascend/ascend-toolkit/latest
export PTCOPY_ENABLE=1
export TASK_QUEUE_ENABLE=1
export DYNAMIC_OP="ADD#MUL"
export COMBINED_ENABLE=1
export DYNAMIC_COMPILE_ENABLE=0
export EXPERIMENTAL_DYNAMIC_PARTITION=0
export ASCEND_GLOBAL_EVENT_ENABLE=0
PORT=29500 ./tools/dist_train.sh configs/retinanet/retinanet_r50_fpn_1x_coco.py 8 --cfg-options optimizer.lr=0.04 --seed 0 --gpu-ids 0 --no-validate --opt-level O1