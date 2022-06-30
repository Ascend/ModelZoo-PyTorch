#!/bin/bash

rm -rf spnasnet_100.onnx

python3.7 pthtar2onnx.py model_best.pth.tar spnasnet_100.onnx

source /usr/local/Ascend/ascend-lastest/set_env.sh

rm -rf spnasnet_100_bs1.om spnasnet_100_bs16.om

atc --model=./spnasnet_100.onnx --framework=5 --output=spnasnet_100_bs1 --input_format=NCHW --input_shape="actual_input_1:1,3,224,224" --log=info --soc_version=Ascend310

atc --model=./spnasnet_100.onnx --framework=5 --output=spnasnet_100_bs16 --input_format=NCHW --input_shape="actual_input_1:16,3,224,224" --log=info --soc_version=Ascend310

if [ -f "spnasnet_100_bs1.om" ] && [ -f "spnasnet_100_bs16.om" ]; then
    echo "success"
else
    echo "fail!"
fi
