#!/bin/bash

atc --model=$1 \
    --output=$2 \
    --input_shape=$3 \
    --log=error \
    --framework=5 \
    --soc_version=Ascend310 \
    --input_format=ND \
    --auto_tune_mode="GA,RL" \
    --op_select_implmode=high_performance
