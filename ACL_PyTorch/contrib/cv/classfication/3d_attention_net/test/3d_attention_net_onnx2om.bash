#!/bin/bash

export install_path=/usr/local/Ascend/ascend-toolkit/latest
export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export PYTHONPATH=${install_path}/atc/python/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=${install_path}/atc/lib64:${install_path}/acllib/lib64:$LD_LIBRARY_PATH
export ASCEND_OPP_PATH=${install_path}/opp
export ASCEND_AICPU_PATH=/usr/local/Ascend/ascend-toolkit/latest
#export ASCEND_SLOG_PRINT_TO_STDOUT=1
#export ASCEND_GLOBAL_LOG_LEVEL=0 #debug 0 --> info 1 --> warning 2 --> error 3
#export DUMP_GE_GRAPH=2

atc --framework=5 --model=3d_attention_net_resize_optimized.onnx --output=3d_attention_net_resize_autotune_optimized_bs1 --input_format=NCHW --input_shape="image:1,3,32,32" --log=debug -soc_version=Ascend310 --auto_tune_mode="RL,GA"

atc --framework=5 --model=3d_attention_net_resize_optimized.onnx --output=3d_attention_net_resize_autotune_optimized_bs16 --input_format=NCHW --input_shape="image:16,3,32,32" --log=debug --soc_version=Ascend310 --auto_tune_mode="RL,GA"