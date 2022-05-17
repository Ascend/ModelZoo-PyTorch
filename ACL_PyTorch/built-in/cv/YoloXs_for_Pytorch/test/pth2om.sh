#!/bin/bash

set -eu

mkdir -p models
rm -rf models/*.onnx
cd YOLOX

python tools/export_onnx.py -c ../yolox_s.pth -f exps/default/yolox_s.py --dynamic

if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

mv yolox.onnx ../models

cd ..
rm -rf models/*.om
source env.sh

atc --model=yolox.onnx --framework=5 --output=yolox --input_format=NCHW --optypelist_for_implmode="Sigmoid" \
 --op_select_implmode=high_performance --input_shape='images:4,3,640,640'  --log=info --soc_version=Ascend710

if [ -f "models/yolox.om" ]; then
    echo "success"
else
    echo "fail!"
fi
