#!/bin/bash

export install_path="/usr/local/Ascend/ascend-toolkit/latest"
export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export PYTHONPATH=${install_path}/atc/python/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=${install_path}/atc/lib64:$LD_LIBRARY_PATH
export ASCEND_OPP_PATH=${install_path}/opp

export ASCEND_SLOG_PRINT_TO_STDOUT=1


${install_path}/atc/bin/atc --framework=5 --output=./i3d_bs1  --input_format=NCHW  --soc_version=Ascend310 \
    --model=./i3d.onnx --input_shape="0:1,30,3,32,256,256" 


