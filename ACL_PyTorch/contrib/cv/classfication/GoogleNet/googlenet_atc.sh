#!/bin/bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
#export GLOBAL_LOG_LEVEL=3 #log level
#export SLOG_PRINT_TO_STDOUT=0  #print log to tenminal
#export DUMP_GE_GRAPH=0 #dump ge

atc --model=./googlenet.onnx --framework=5 --output=googlenet_bs16 --input_format=NCHW --input_shape="actual_input_1:16,3,224,224" --log=info --soc_version=Ascend310

#atc --model=./googlenet.onnx --framework=5 --output=googlenet_bs1 --input_format=NCHW --input_shape="actual_input_1:1,3,224,224" --log=info --soc_version=Ascend310 --insert_op_conf=aipp_googlenet_pth.config
