#! /bin/bash
source test/env_npu.sh
export HCCL_CONNECT_TIMEOUT=6000

GPUS_PER_NODE=1
# Change for multinode config

MASTER_ADDR=localhost
MASTER_PORT=12346
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
OPTIONS="run.max_epoch=2 run.iters_per_epoch=240 run.batch_size_train=10 run.batch_size_eval=10 "
torchrun $DISTRIBUTED_ARGS train.py --cfg-path train_configs/minigpt4_stage2_finetune.yaml --options ${OPTIONS} 
