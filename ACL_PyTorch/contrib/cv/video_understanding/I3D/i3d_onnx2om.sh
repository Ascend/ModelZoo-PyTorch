#!/bin/bash

${install_path}/atc/bin/atc --framework=5 --output=./i3d_nl_dot_bs1  --input_format=NCHW  --soc_version=Ascend310 \
    --model=./i3d_nl_dot.onnx --input_shape="0:1,10,3,32,256,256"


