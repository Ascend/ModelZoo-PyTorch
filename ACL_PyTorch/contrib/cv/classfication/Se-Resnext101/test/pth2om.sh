#!/bin/bash

rm -rf se-resnext101.onnx
python3.7 se_resnext101_pth2onnx.py state_dict.pth se-resnext101.onnx
source env.sh
rm -rf se-resnext101_bs1.om se-resnext101_bs16.om
atc --framework=5 --model=./se-resnext101.onnx --input_format=NCHW --input_shape="image:1,3,224,224" --output=se-resnext101_bs1 --log=debug --soc_version=Ascend310
atc --framework=5 --model=./se-resnext101.onnx --input_format=NCHW --input_shape="image:16,3,224,224" --output=se-resnext101_bs16 --log=debug --soc_version=Ascend310
if [ -f "se-resnext101_bs1.om" ] && [ -f "se-resnext101_bs16.om" ]; then
    echo "success"
else
    echo "fail!"
fi