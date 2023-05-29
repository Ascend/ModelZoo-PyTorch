#!/bin/bash

TOTAL_NUM_UPDATES=20935	  
WARMUP_UPDATES=1256      
LR=2e-05                
NUM_CLASSES=2
MAX_SENTENCES=64       
ROBERTA_PATH=./pre_train_model/roberta.base/model.pt
DISTRIBUTED_BACKEND=hccl
DISTRIBUTED_WORLD_SIZE=8
OUTPUT_DIR=./output

if [ ! -d $OUTPUT_DIR ]
then
    mkdir $OUTPUT_DIR
fi

RANK_ID_START=0
RANK_SIZE=8

for((RANK_ID=$RANK_ID_START;RANK_ID<$((RANK_SIZE+RANK_ID_START));RANK_ID++));
do

KERNEL_NUM=$(($(nproc)/8))
PID_START=$((KERNEL_NUM * RANK_ID))
PID_END=$((PID_START + KERNEL_NUM - 1))

nohup \
taskset -c $PID_START-$PID_END python3 -u ./train.py ./data/SST-2/ "$@"  \
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
    --use-apex --use-npu-adam --opt-level O2 --loss-scale 32 \
    --max-epoch 10 \
    --log-interval 10 \
    --find-unused-parameters \
    --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
    --distributed-backend $DISTRIBUTED_BACKEND \
    --distributed-world-size $RANK_SIZE \
    --distributed-no-spawn \
    --npu \
    --no-save \
    --num-workers $(($(nproc)/8)) \
    --load-checkpoint-on-all-dp-ranks \
    --device-id $RANK_ID \
	--distributed-rank $RANK_ID & 
done

wait

echo "Train finish, check the log file: ./output/8p_npu_log"
mv nohup.out ./output/8p_npu_log