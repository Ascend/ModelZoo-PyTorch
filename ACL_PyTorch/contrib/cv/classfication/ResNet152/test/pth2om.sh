#!/bin/bash

rm -rf resnet152.onnx
python3.7 resnet152_pth2onnx.py resnet152-b121ed2d.pth resnet152.onnx
source /usr/local/Ascend/ascend-toolkit/set_env.sh
rm -rf resnet152_bs1.om resnet152_bs16.om
atc --framework=5 --model=./resnet152.onnx --input_format=NCHW --input_shape="image:1,3,224,224" --output=resnet152_bs1 --log=debug --soc_version=Ascend310
atc --framework=5 --model=./resnet152.onnx --input_format=NCHW --input_shape="image:16,3,224,224" --output=resnet152_bs16 --log=debug --soc_version=Ascend310
if [ -f "resnet152_bs1.om" ] && [ -f "resnet152_bs16.om" ]; then
    echo "success"
else
    echo "fail!"
fi