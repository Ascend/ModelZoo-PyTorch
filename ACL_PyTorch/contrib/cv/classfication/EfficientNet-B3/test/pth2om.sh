#!/bin/bash

source env.sh

rm -rf efficientnetB3*.onnx
rm -rf efficientnetB3*.om 

python3.7 efficientnetB3_pth2onnx.py EN-B3_dds_8gpu.pyth ./pycls/configs/dds_baselines/effnet/EN-B3_dds_8gpu.yaml efficientnetB3.onnx 

atc --framework=5 --model=./efficientnetB3.onnx --input_format=NCHW --input_shape="image:1,3,300,300" --output=efficientnetB3_bs1 --log=debug --soc_version=Ascend310

atc --framework=5 --model=./efficientnetB3.onnx --input_format=NCHW --input_shape="image:16,3,300,300" --output=efficientnetB3_bs16 --log=debug --soc_version=Ascend310 --auto_tune_mode="RL,GA"

if [ -f "efficientnetB3_bs1.om" ] && [ -f "efficientnetB3_bs16.om" ]; then
    echo "success"
else
    echo "fail!"
fi