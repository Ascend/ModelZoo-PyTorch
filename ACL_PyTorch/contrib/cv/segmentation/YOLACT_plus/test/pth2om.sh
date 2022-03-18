#!/bin/bash
pth_path=weights/yolact_plus_resnet50_54_800000.pth
onnx_name=yolact_plus

python3.7.5 pth2onnx.py --trained_model=$pth_path --outputName=$onnx_name --dynamic=$dynamic

source env.sh

onnx_full_name=$2".onnx"

${install_path}/atc/bin/atc --framework=5 --output=yolact_plus_bs1  --input_format=NCHW  --soc_version=Ascend310 --model=$onnx_full_name --input_shape="input.1:1,3,550,550"
${install_path}/atc/bin/atc --framework=5 --output=yolact_plus_bs8  --input_format=NCHW  --soc_version=Ascend310 --model=$onnx_full_name --input_shape="input.1:8,3,550,550"
