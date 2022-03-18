#!/bin/bash

export install_path="/usr/local/Ascend/ascend-toolkit/latest"
export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export PYTHONPATH=${install_path}/atc/python/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=${install_path}/atc/lib64:${install_path}/acllib/lib64:$LD_LIBRARY_PATH
export ASCEND_OPP_PATH=${install_path}/opp

atc --model=$1 \
    --framework=5 \
    --output=$2 \
    --log=error \
    --soc_version=Ascend310 \
    --input_shape="feats:4,64,4000;feat_lens:4" \
    --input_fp16_nodes="feats" \
    --out_nodes="Cast_3388:0;LogSoftmax_3393:0" \
    --output_type="LogSoftmax_3393:0:FP16"
