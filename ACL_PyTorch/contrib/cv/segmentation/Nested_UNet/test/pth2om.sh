#!/bin/bash

rm -rf nested_unet.onnx
python3.7 nested_unet_pth2onnx.py nested_unet.pth nested_unet.onnx
source env.sh
rm -rf nested_unet_bs1.om nested_unet_bs16.om
atc --model=./nested_unet.onnx --framework=5 --auto_tune_mode="GA,RL" --output=nested_unet_bs1 --input_format=NCHW --input_shape="actual_input_1:1,3,96,96" --log=info --soc_version=Ascend310
atc --model=./nested_unet.onnx --framework=5 --auto_tune_mode="GA,RL" --output=nested_unet_bs16 --input_format=NCHW --input_shape="actual_input_1:16,3,96,96" --log=info --soc_version=Ascend310
if [ -f "nested_unet_bs1.om" ] && [ -f "nested_unet_bs16.om" ]; then
    echo "success"
else
    echo "fail!"
fi