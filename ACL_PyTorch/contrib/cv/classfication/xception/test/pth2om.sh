#!/bin/bash

rm -rf xception.onnx
python3.7 xception_pth2onnx.py xception-c0a72b38.pth.tar xception.onnx
source env.sh
rm -rf xception_bs1.om xception_bs16.om
atc --framework=5 --model=xception.onnx --output=xception_bs1 --input_format=NCHW --input_shape="image:1,3,299,299" --log=debug --soc_version=Ascend310
atc --framework=5 --model=xception.onnx --output=xception_bs16 --input_format=NCHW --input_shape="image:16,3,299,299" --log=debug --soc_version=Ascend310
if [ -f "xception_bs1.om" ] && [ -f "xception_bs16.om" ]; then
    echo "success"
else
    echo "fail!"
fi