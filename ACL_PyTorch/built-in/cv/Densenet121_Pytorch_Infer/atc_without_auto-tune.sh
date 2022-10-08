#!/bin/bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh

export SLOG_PRINT_TO_STDOUT=1

atc --model=$1 --framework=5 --output=./resnet_official --input_format=NCHW --input_shape="actual_input_1:16,3,224,224" --enable_small_channel=1 --log=error --soc_version=Ascend310 --insert_op_conf=aipp_TorchVision.config