#!/bin/bash

# please ensure resnest has already installed. "pip install resnest --pre"
source env.sh

rm -rf resnest50.onnx resnest50_sim.onnx
python3.7 resnest_pth2onnx.py --source="./resnest50.pth" --target="resnest50.onnx"
python3.7 -m onnxsim --input-shape="1,3,224,224" --dynamic-input-shape resnest50.onnx resnesst50_sim.onnx

rm  -rf resnest50_b1.om resnest50_b4.om resnest50_b8.om resnest50_b16.om resnest50_b32.om
export REPEAT_TUNE=True
atc --framework=5 --model=./resnest50_sim.onnx --output=resnest50_b1 --input_format=NCHW --input_shape="actual_input_1:1,3,224,224" --log=debug --soc_version=Ascend310 --auto_tune_mode="RL,GA" --op_select_implmode=high_performance --input_fp16_nodes="actual_input_1"
atc --framework=5 --model=./resnest50_sim.onnx --output=resnest50_b16 --input_format=NCHW --input_shape="actual_input_1:16,3,224,224" --log=debug --soc_version=Ascend310 --auto_tune_mode="RL,GA" --op_select_implmode=high_performance --input_fp16_nodes="actual_input_1"

if [ -f "resnest50_b1.om" ] && [ -f "resnest50_b16.om" ]
then
    echo "success"
else
    echo "fail!"
fi
