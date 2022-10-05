#!/bin/bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
#export GLOBAL_LOG_LEVEL=3 #log level
#export SLOG_PRINT_TO_STDOUT=0  #print log to tenminal
#export DUMP_GE_GRAPH=0 #dump ge

atc --model=./inceptionv3.onnx --framework=5 --output=inceptionv3_bs8 --input_format=NCHW --input_shape="actual_input_1:8,3,299,299" --log=info --soc_version=$1 --insert_op_conf=aipp_inceptionv3_pth.config

