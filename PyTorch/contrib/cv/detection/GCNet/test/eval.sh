# PORT=29888 ./tools/dist_train.sh ./configs/gcnet/mask_rcnn_r50_fpn_r4_gcb_c3-c5_1x_coco.py 8 --npu-ids 0 --cfg-options optimizer.lr=0.02 --seed 0 --opt-level O1 --loss-scale 16.0
source ./test//env_npu.sh
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

data_path=""
weight_path=""

python ./tools/test.py ./configs/gcnet/mask_rcnn_r50_fpn_r4_gcb_c3-c5_1x_coco.py $weight_path --cfg-options data_root=$data_path --eval bbox segm proposal
