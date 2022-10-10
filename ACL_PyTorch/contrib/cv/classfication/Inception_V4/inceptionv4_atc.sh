#!/bin/bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
batch_size=$1
chip_name=$2

atc --model=./inceptionv4.onnx --framework=5 --output=inceptionv4_bs${batch_size} --input_format=NCHW --input_shape="actual_input_1:${batch_size},3,299,299" --log=info --soc_version=Ascend${chip_name}
