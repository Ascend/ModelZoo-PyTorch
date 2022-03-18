#!/bin/bash
export install_path=/usr/local/Ascend/ascend-toolkit/latest
export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export PYTHONPATH=${install_path}/atc/python/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=${install_path}/atc/lib64:${install_path}/acllib/lib64:$LD_LIBRARY_PATH
export ASCEND_OPP_PATH=${install_path}/opp

export SLOG_PRINT_TO_STDOUT=1

/usr/local/Ascend/ascend-toolkit/latest/atc/bin/atc --model=$1 --framework=5 --output=./resnet_official --input_format=NCHW --input_shape="actual_input_1:16,3,224,224" --enable_small_channel=1 --log=error --soc_version=Ascend310 --insert_op_conf=aipp_TorchVision.config