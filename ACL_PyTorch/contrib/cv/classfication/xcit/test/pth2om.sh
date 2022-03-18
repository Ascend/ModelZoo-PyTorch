#!/bin/bash
source env.sh
rm -rf xcit_b1.onnx
mkdir onnx_models
python3.7 xcit_pth2onnx.py --pretrained=pretrained/xcit_small_12_p16_224.pth
python3.7 xcit_pth2onnx.py --pretrained=pretrained/xcit_small_12_p16_224.pth --batch-size=16
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
atc --framework=5 --model=onnx_models/xcit_b1.onnx  --output=xcit_b1 --input_format=NCHW --input_shape="image:1,3,224,224" --log=debug --soc_version=Ascend310
if [ -f "xcit_b1.om" ] && [ -f "xcit_b1.om" ]; then
    echo "success"
else
    echo "fail!"
fi
atc --framework=5 --model=onnx_models/xcit_b16.onnx  --output=xcit_b16 --input_format=NCHW --input_shape="image:16,3,224,224" --log=debug --soc_version=Ascend310 --auto_tune_mode="RL,GA"
if [ -f "xcit_b16.om" ] && [ -f "xcit_b16.om" ]; then
    echo "success"
else
    echo "fail!"
fi