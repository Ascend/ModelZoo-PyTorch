#!/bin/bash

rm -rf resnet101.onnx
python3.7 resnet101_pth2onnx.py resnet101-63fe2227.pth resnet101.onnx
source /usr/local/Ascend/ascend-toolkit/set_env.sh
rm -rf resnet101_bs1.om resnet101_bs16.om
atc --framework=5 --model=./resnet101.onnx --input_format=NCHW --input_shape="image:1,3,224,224" --output=resnet101_bs1 --log=debug --soc_version=Ascend310
atc --framework=5 --model=./resnet101.onnx --input_format=NCHW --input_shape="image:16,3,224,224" --output=resnet101_bs16 --log=debug --soc_version=Ascend310
if [ -f "resnet101_bs1.om" ] && [ -f "resnet101_bs16.om" ]; then
    echo "success"
else
    echo "fail!"
fi