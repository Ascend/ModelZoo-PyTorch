#!/bin/bash

# pth2onnx
rm -rf deit_small_patch16_224.onnx
python3.7 deit_small_pth2onnx.py deit_small_patch16_224-cd65a155.pth deit_small_patch16_224_onnx.onnx

python3.7 -m onnxsim --input-shape="8,3,224,224" deit_small_patch16_224_onnx.onnx deit_small_patch16_224_bs8_onnxsim.onnx
python3.7 -m onnxsim --input-shape="1,3,224,224" deit_small_patch16_224_onnx.onnx deit_small_patch16_224_bs1_onnxsim.onnx

# onnx2om
source /usr/local/Ascend/ascend-toolkit/set_env.sh
rm -rf deit_small_bs1.om deit_small_bs8.om
atc --framework=5 --model=./deit_small_patch16_224_bs1_onnxsim.onnx --input_format=NCHW --input_shape="image:1,3,224,224" --output=deit_small_bs1 --log=debug --soc_version=Ascend310
atc --framework=5 --model=./deit_small_patch16_224_bs8_onnxsim.onnx --input_format=NCHW --input_shape="image:8,3,224,224" --output=deit_small_bs16 --log=debug --soc_version=Ascend310

if [ -f "deit_small_bs1.om" ] && [ -f "deit_small_bs8.om" ]; then
    echo "success"
else
    echo "fail!"
fi