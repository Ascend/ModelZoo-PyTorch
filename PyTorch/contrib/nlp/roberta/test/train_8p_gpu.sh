#!/bin/bash

TOTAL_NUM_UPDATES=20935	  
WARMUP_UPDATES=1256      
LR=1e-05                
NUM_CLASSES=2
MAX_SENTENCES=32       
ROBERTA_PATH=./pre_train_model/roberta.base/model.pt
DISTRIBUTED_WORLD_SIZE=8
DISTRIBUTED_BACKEND=nccl
OUTPUT_DIR=./output

if [ ! -d $OUTPUT_DIR ]
then
    mkdir $OUTPUT_DIR
fi

python -u ./train.py ./data/SST-2/ \
    --restore-file $ROBERTA_PATH \
    --max-positions 512 \
    --batch-size $MAX_SENTENCES \
    --max-tokens 4400 \
    --pad-length 70 \
    --task sentence_prediction \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --init-token 0 --separator-token 2 \
    --arch roberta_base \
    --criterion sentence_prediction \
    --num-classes $NUM_CLASSES \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.1 --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-06 \
    --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    --use-apex --opt-level O2 --loss-scale 64 \
    --max-epoch 10 \
    --log-interval 10 \
    --find-unused-parameters \
    --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
    --distributed-world-size $DISTRIBUTED_WORLD_SIZE \
    --distributed-backend $DISTRIBUTED_BACKEND \
    --gpu \
    --no-save \
    --no-progress-bar
