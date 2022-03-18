#!/bin/bash

rm -rf squeezenet1_1.onnx
python3.7 squeezenet1_1_pth2onnx.py squeezenet1_1-f364aa15.pth squeezenet1_1.onnx
source env.sh
rm -rf squeezenet1_1_bs1.om squeezenet1_1_bs16.om
atc --framework=5 --model=./squeezenet1_1.onnx --input_format=NCHW --input_shape="image:1,3,224,224" --output=squeezenet1_1_bs1 --log=debug --soc_version=Ascend310
atc --framework=5 --model=./squeezenet1_1.onnx --input_format=NCHW --input_shape="image:16,3,224,224" --output=squeezenet1_1_bs16 --log=debug --soc_version=Ascend310 --auto_tune_mode="RL,GA"
if [ -f "squeezenet1_1_bs1.om" ] && [ -f "squeezenet1_1_bs16.om" ]; then
    echo "success"
else
    echo "fail!"
fi