#!/bin/bash
export install_path=/usr/local/Ascend/ascend-toolkit/latest
export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export PYTHONPATH=${install_path}/atc/python/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=${install_path}/atc/lib64:${install_path}/acllib/lib64:$LD_LIBRARY_PATH
export ASCEND_OPP_PATH=${install_path}/opp
#export GLOBAL_LOG_LEVEL=3 #log level
#export SLOG_PRINT_TO_STDOUT=0  #print log to tenminal
#export DUMP_GE_GRAPH=0 #dump ge

atc --model=./inceptionv4.onnx --framework=5 --output=inceptionv4_bs16 --input_format=NCHW --input_shape="actual_input_1:16,3,299,299" --log=info --soc_version=Ascend310

#atc --model=./inceptionv4.onnx --framework=5 --output=inceptionv4_bs1 --input_format=NCHW --input_shape="actual_input_1:1,3,300,300" --log=info --soc_version=Ascend310 --insert_op_conf=aipp_inceptionv4_pth.config
