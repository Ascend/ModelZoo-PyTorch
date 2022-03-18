#!/bin/bash
source env.sh
python3.7 pytorch2onnx.py mmaction2/configs/recognition/tsm/tsm_k400_pretrained_r50_1x1x8_25e_ucf101_rgb.py ./tsm_k400_pretrained_r50_1x1x8_25e_ucf101_rgb_20210630-1fae312b.pth --output-file=tsm.onnx --softmax --verify --show --shape 1 8 3 224 224

mkdir -p onnx_sim
python3.7 -m onnxsim --input-shape="1,8,3,224,224" tsm.onnx onnx_sim/tsm_bs1.onnx
python3.7 -m onnxsim --input-shape="16,8,3,224,224" tsm.onnx onnx_sim/tsm_bs16.onnx

mkdir -p om
atc --model=onnx_sim/tsm_bs1.onnx --framework=5 --output=om/tsm_bs1 --input_format=NCDHW --input_shape="video:1,8,3,224,224" --log=debug --soc_version=Ascend310 --auto_tune_mode="RL,GA"
atc --model=onnx_sim/tsm_bs16.onnx --framework=5 --output=om/tsm_bs16 --input_format=NCDHW --input_shape="video:16,8,3,224,224" --log=debug --soc_version=Ascend310 --auto_tune_mode="RL,GA"
if [ -f "om/tsm_bs1.om" ] && [ -f "om/tsm_bs16.om" ]; then
    echo "success"
else
    echo "fail!"
fi