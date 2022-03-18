#!/bin/bash

rm -rf res2net101_v1b.onnx
python3.7 res2net101_v1b_pth2onnx.py res2net101_v1b_26w_4s-0812c246.pth res2net101_v1b.onnx
source env.sh
rm -rf res2net101_v1b_bs1.om res2net101_v1b_bs16.om
atc --framework=5 --model=./res2net101_v1b.onnx --input_format=NCHW --input_shape="image:1,3,224,224" --output=res2net101_v1b_bs1 --log=debug --soc_version=Ascend310
atc --framework=5 --model=./res2net101_v1b.onnx --input_format=NCHW --input_shape="image:16,3,224,224" --output=res2net101_v1b_bs16 --log=debug --soc_version=Ascend310
if [ -f "res2net101_v1b_bs1.om" ] && [ -f "res2net101_v1b_bs16.om" ]; then
    echo "success"
else
    echo "fail!"
fi