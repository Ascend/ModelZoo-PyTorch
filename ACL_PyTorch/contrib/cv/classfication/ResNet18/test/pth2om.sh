#!/bin/bash

rm -rf resnet18.onnx
python3.7 resnet18_pth2onnx.py resnet18-f37072fd.pth resnet18.onnx
source env.sh
rm -rf resnet18_bs1.om resnet18_bs16.om
atc --framework=5 --model=./resnet18.onnx --input_format=NCHW --input_shape="image:1,3,224,224" --output=resnet18_bs1 --log=debug --soc_version=Ascend310
atc --framework=5 --model=./resnet18.onnx --input_format=NCHW --input_shape="image:16,3,224,224" --output=resnet18_bs16 --log=debug --soc_version=Ascend310
if [ -f "resnet18_bs1.om" ] && [ -f "resnet18_bs16.om" ]; then
    echo "success"
else
    echo "fail!"
fi