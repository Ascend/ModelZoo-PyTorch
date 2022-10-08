#!/bin/bash

source /usr/local/Ascend/ascend-toolkit/set_env.sh

atc --model=$1 \
    --framework=5 \
    --output=$2 \
    --log=error \
    --soc_version=Ascend310 \
    --input_shape="feats:4,64,4000;feat_lens:4" \
    --input_fp16_nodes="feats" \
    --out_nodes="Cast_3388:0;LogSoftmax_3393:0" \
    --output_type="LogSoftmax_3393:0:FP16"
