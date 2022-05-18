#!/bin/bash

set -eu

mkdir -p models
rm -rf models/*.onnx

python tools/export_onnx.py -c ./yolox_s.pth -f exps/default/yolox_s.py --dynamic

if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

rm -rf models/*.om

atc --model=yolox.onnx --framework=5 --output=./models/yolox --input_format=NCHW --optypelist_for_implmode="Sigmoid" \
 --op_select_implmode=high_performance --input_shape='images:4,3,640,640'  --log=info --soc_version=Ascend710

if [ -f "models/yolox.om" ]; then
    echo "success"
else
    echo "fail!"
fi
