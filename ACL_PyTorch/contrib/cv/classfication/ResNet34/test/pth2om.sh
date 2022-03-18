#!/bin/bash

rm -rf resnet34.onnx
python3.7 resnet34_pth2onnx.py resnet34-b627a593.pth resnet34.onnx
source env.sh
rm -rf resnet34_bs1.om resnet34_bs16.om
atc --framework=5 --model=./resnet34.onnx --input_format=NCHW --input_shape="image:1,3,224,224" --output=resnet34_bs1 --log=debug --soc_version=Ascend310
atc --framework=5 --model=./resnet34.onnx --input_format=NCHW --input_shape="image:16,3,224,224" --output=resnet34_bs16 --log=debug --soc_version=Ascend310
if [ -f "resnet34_bs1.om" ] && [ -f "resnet34_bs16.om" ]; then
    echo "success"
else
    echo "fail!"
fi