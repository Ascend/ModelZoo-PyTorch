#!/bin/bash

rm -rf vnet.onnx
python3.7 vnet_pth2onnx.py vnet_model_best.pth.tar vnet.onnx
source /usr/local/Ascend/ascend-toolkit/set_env.sh
rm -rf vnet_bs1.om vnet_bs16.om
atc --model=./vnet.onnx --framework=5 --output=vnet_bs1 --input_format=NCDHW --input_shape="actual_input_1:1,1,64,80,80" --log=info --soc_version=Ascend310
atc --model=./vnet.onnx --framework=5 --output=vnet_bs16 --input_format=NCDHW --input_shape="actual_input_1:16,1,64,80,80" --log=info --soc_version=Ascend310

if [ -f "vnet_bs1.om" ] && [ -f "vnet_bs16.om" ]; then
    echo "success"
else
    echo "fail!"
fi