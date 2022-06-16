#!/bin/bash
soc=$1
onnx=$2
om=$3
bs=$4


atc --model=${onnx}.onnx \
    --output=${om} \
    --input_shape="input:${bs},64600" \
    --log=error \
    --framework=5 \
    --soc_version=${soc} \
    --input_format=ND \
    --optypelist_for_implmode="Sigmoid" \
    --op_select_implmode=high_performance

