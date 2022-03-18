#!/bin/bash

DATA_DIR="/data/chid/preprocessed/"
CHECKPOINT_PATH="/data/checkpoints/CPM-medium"
RESULTS_DIR="results/"
MODEL_NAME="zeroshot-test"
TOKENIZER_PATH="bpe_3w_new/"
MPSIZE=1
NLAYERS=24
NHIDDEN=1024
NATT=16
MAXSEQLEN=1024

CUR_PATH=$(realpath $0)
CUR_DIR=$(dirname ${CUR_PATH})
DS_CONFIG="${CUR_DIR}/ds_finetune_medium.json"

python3 -m torch.distributed.launch --master_port ${1-1122} --nproc_per_node 4 zero-shot_chid.py \
       --data_dir ${DATA_DIR} \
       --model-parallel-size ${MPSIZE} \
       --num-layers ${NLAYERS} \
       --hidden-size ${NHIDDEN} \
       --load ${CHECKPOINT_PATH} \
       --num-attention-heads ${NATT} \
       --seq-length ${MAXSEQLEN} \
       --max-position-embeddings 1024 \
       --tokenizer-type GPT2BPETokenizer \
       --fp16 \
       --out-seq-length 512 \
       --tokenizer-path ${TOKENIZER_PATH} \
       --vocab-size 30000 \
       --batch-size 2 \
       --deepspeed \
       --deepspeed_config ${DS_CONFIG} \
       --seed 23333 \
       --results_dir ${RESULTS_DIR} \
       --model_name ${MODEL_NAME} \
