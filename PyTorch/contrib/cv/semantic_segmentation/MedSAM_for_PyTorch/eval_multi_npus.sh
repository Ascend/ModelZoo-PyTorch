#!/bin/bash

TASK="MedSAM-ViT-B"

NNODES=1
NODE_RANK=0
GPUS_PER_NODE=1
MASTER_ADDR=localhost
MASTER_PORT=6789

python eval_multi_npus.py \
    --task_name "${TASK}" \
    --model_type vit_b \
    --data_path ../datasets/npy/CT_Abd \
    --checkpoint ../models/sam_vit_b_01ec64.pth \
    --work_dir ./work_dir \
    --num_workers 8 \
    --nnodes "${NNODES}" \
    --node_rank "${NODE_RANK}" \
    --nproc_per_node "${GPUS_PER_NODE}" \
    --node_rank ${NODE_RANK} \
    --init_method tcp://${MASTER_ADDR}:${MASTER_PORT}


wait ## Wait for the tasks on nodes to finish
echo "END TIME: $(date)"
