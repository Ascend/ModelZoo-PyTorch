#!/bin/bash

OUTPUT_DIR=./output
ROBERTA_PATH=${OUTPUT_DIR}/checkpoints/checkpoint_best.pt
VALID_SPLIT=valid

if [ ! -d $OUTPUT_DIR ]
then
    mkdir $OUTPUT_DIR
fi


python3 -u ./fairseq_cli/validate.py ./data/SST-2/ \
    --valid-subset $VALID_SPLIT \
    --path $ROBERTA_PATH \
    --batch-size 64 \
    --task sentence_prediction \
    --criterion sentence_prediction \
    --device-id 0 \
    --pad-length 70 \
    --fp16 \
    --npu > $OUTPUT_DIR/npu_eval.log