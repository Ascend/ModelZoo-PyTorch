#!/bin/bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh

atc --model=./resnext50.onnx --framework=5 --output=resnext50_bs16 --input_format=NCHW --input_shape="actual_input_1:16,3,224,224" --log=info --soc_version=Ascend310

#atc --model=./resnext50.onnx --framework=5 --output=resnext50_bs1 --input_format=NCHW --input_shape="actual_input_1:1,3,224,224" --log=info --soc_version=Ascend310 --insert_op_conf=aipp_resnext50_pth.config
