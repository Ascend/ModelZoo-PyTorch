#!/bin/bash

rm -rf ICNet.onnx
PTH_NAME=rankid0_icnet_resnet50_192_0.687_best_model.pth
python3.7 ICNet_pth2onnx.py $PTH_NAME ICNet.onnx

source /usr/local/Ascend/ascend-toolkit/set_env.sh

rm -rf ICNet_bs1.om ICNet_bs4.om
atc --framework=5 --model=./ICNet.onnx --output=ICNet_bs1 --out_nodes="Resize_317:0" --input_format=NCHW --input_shape="actual_input_1: 1,3,1024,2048" --log=debug --soc_version=Ascend310

atc --framework=5 --model=./ICNet.onnx --output=ICNet_bs4 --out_nodes="Resize_317:0" --input_format=NCHW --input_shape="actual_input_1: 4,3,1024,2048" --log=debug --soc_version=Ascend310

if [ -f "ICNet_bs1.om" ] && [ -f "ICNet_bs4.om" ]; then
    echo "success"
else
    echo "fail!"
fi