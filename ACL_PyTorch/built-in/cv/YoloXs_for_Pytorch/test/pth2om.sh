#!/bin/bash

set -eu

mkdir -p models
rm -rf models/*.onnx
cd YOLOX

python tools/export_onnx.py --output-name yolox_x_bs1.onnx -n yolox-x -c ../yolox_x.pth

if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

mv yolox_x_bs1.onnx ../models

cd ..
rm -rf models/*.om
source env.sh

atc --model=yolox_x_fix.onnx --framework=5 --output=yolox_x_fix --input_format=NCHW --input_shape='images:1,3,640,640' --log=info --soc_version=Ascend310 --out_nodes="Conv_498:0;Conv_499:0;Conv_491:0;Conv_519:0;Conv_520:0;Conv_512:0;Conv_540:0;Conv_541:0;Conv_533:0"

if [ -f "models/yolox_x_bs1.om" ]; then
    echo "success"
else
    echo "fail!"
fi
