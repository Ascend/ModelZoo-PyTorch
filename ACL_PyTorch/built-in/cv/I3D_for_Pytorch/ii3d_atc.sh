#!/bin/bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh

export ASCEND_SLOG_PRINT_TO_STDOUT=1


atc --framework=5 --output=./i3d_bs1  --input_format=NCHW  --soc_version=Ascend310 \
    --model=./i3d.onnx --input_shape="0:1,30,3,32,256,256" 


