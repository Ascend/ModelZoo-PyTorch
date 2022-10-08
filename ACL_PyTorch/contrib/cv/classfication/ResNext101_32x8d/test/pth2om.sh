#!/bin/bash

rm -rf resnext101_32x8.onnx
python3.7 resnext101_32x8d_pth2onnx.py resnext101_32x8d-8ba56ff5.pth resnext101_32x8d.onnx
source /usr/local/Ascend/ascend-toolkit/set_env.sh
rm -rf resnext101_32x8d_bs1.om resnext101_32x8d_bs16.om
atc --framework=5 --model=./resnext101_32x8d.onnx --input_format=NCHW --input_shape="image:1,3,224,224" --output=resnext101_32x8d_bs1 --log=debug --soc_version=Ascend310
atc --framework=5 --model=./resnext101_32x8d.onnx --input_format=NCHW --input_shape="image:16,3,224,224" --output=resnext101_32x8d_bs16 --log=debug --soc_version=Ascend310
if [ -f "resnext101_32x8d_bs1.om" ] && [ -f "resnext101_32x8d_bs16.om" ]; then
    echo "success"
else
    echo "fail!"
fi