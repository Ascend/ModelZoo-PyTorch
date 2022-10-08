#!/bin/bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh

mkdir -p onnx_sim
mkdir -p om

python pytorch2onnx.py mmaction2/configs/recognition/slowfast/slowfast_r50_8x8x1_256e_kinetics400_rgb.py ./slowfast_r50_8x8x1_256e_kinetics400_rgb_20200716-73547d2b.pth --output-file=slowfast.onnx --softmax --verify --show --shape 1 1 3 32 224 224

python3.7 -m onnxsim --input-shape="1,1,3,32,224,224" slowfast.onnx onnx_sim/slowfast_bs1.onnx
python3.7 -m onnxsim --input-shape="16,1,3,32,224,224" slowfast.onnx onnx_sim/slowfast_bs16.onnx

atc --model=onnx_sim/slowfast_bs1.onnx --framework=5 --output=om/slowfast_bs1 --input_format=ND --input_shape="video:1,1,3,32,224,224" --log=debug --soc_version=Ascend310 --auto_tune_mode="RL,GA"
atc --model=onnx_sim/slowfast_bs16.onnx --framework=5 --output=om/slowfast_bs16 --input_format=ND --input_shape="video:16,1,3,32,224,224" --log=debug --soc_version=Ascend310 --auto_tune_mode="RL,GA"

if [ -f "om/slowfast_bs1.om" ] && [ -f "om/slowfast_bs16.om" ]; then
    echo "success"
else
    echo "fail!"
fi