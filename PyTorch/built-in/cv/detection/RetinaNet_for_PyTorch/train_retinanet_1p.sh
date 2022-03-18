source pt_set_env.sh
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_LOG_LEVEL=3
export PTCOPY_ENABLE=1
export TASK_QUEUE_ENABLE=1
export DYNAMIC_OP="ADD#MUL"
export COMBINED_ENABLE=1
export DYNAMIC_COMPILE_ENABLE=0
export EXPERIMENTAL_DYNAMIC_PARTITION=0
export ASCEND_GLOBAL_EVENT_ENABLE=0
export NPUID=0
export HCCL_WHITELIST_DISABLE=1
export SCALAR_TO_HOST_MEM=1

export RANK=0
python3.7 ./tools/train.py configs/retinanet/retinanet_r50_fpn_1x_coco.py \
	--cfg-options \
	optimizer.lr=0.005 \
	--seed 0 \
	--gpu-ids 0 \
	--opt-level O1 &