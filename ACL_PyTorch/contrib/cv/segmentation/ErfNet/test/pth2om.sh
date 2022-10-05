#!/bin/bash

rm -rf ErfNet_origin.onnx
rm -rf ErfNet.onnx
python ErfNet_pth2onnx.py erfnet_pretrained.pth ErfNet_origin.onnx
python modify_bn_weights.py ErfNet_origin.onnx ErfNet.onnx
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

rm -rf ErfNet_bs1.om ErfNet_bs16.om
source /usr/local/Ascend/ascend-toolkit/set_env.sh
atc --framework=5 --model=ErfNet.onnx --output=ErfNet_bs1 --input_format=NCHW --input_shape="actual_input_1:1,3,512,1024" --log=debug --soc_version=Ascend310
atc --framework=5 --model=ErfNet.onnx --output=ErfNet_bs16 --input_format=NCHW --input_shape="actual_input_1:16,3,512,1024" --log=debug --soc_version=Ascend310

if [ -f "ErfNet_bs1.om" ] && [ -f "ErfNet_bs16.om" ]; then
    echo "success"
else
    echo "fail!"
fi
