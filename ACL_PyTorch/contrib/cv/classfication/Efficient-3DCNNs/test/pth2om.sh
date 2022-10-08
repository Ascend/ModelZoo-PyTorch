#!/bin/bash

rm -rf Efficient-3DCNNs.onnx Efficient-3DCNNs_sim.onnx
python3.7 Efficient-3DCNNs_pth2onnx.py ucf101_mobilenetv2_1.0x_RGB_16_best.pth Efficient-3DCNNs.onnx
python3.7 -m onnxsim Efficient-3DCNNs.onnx Efficient-3DCNNs_sim.onnx --input-shape "16,3,16,112,112" --dynamic-input-shape
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

rm -rf Efficient-3DCNNs_bs1.om Efficient-3DCNNs_bs16.om

source /usr/local/Ascend/ascend-toolkit/set_env.sh

atc --framework=5 --model=Efficient-3DCNNs_sim.onnx --output=Efficient-3DCNNs_bs1 --input_format=NCHW --input_shape="image:1,3,16,112,112" --log=error --soc_version=Ascend310
atc --framework=5 --model=Efficient-3DCNNs_sim.onnx --output=Efficient-3DCNNs_bs16 --input_format=NCHW --input_shape="image:16,3,16,112,112" --log=error --soc_version=Ascend310

if [ -f "Efficient-3DCNNs_bs1.om" ] && [ -f "Efficient-3DCNNs_bs16.om" ]; then
    echo "success"
else
    echo "fail!"
fi