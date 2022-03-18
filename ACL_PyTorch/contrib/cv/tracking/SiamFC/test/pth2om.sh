#!/bin/bash

# generate onnx
rm -rf onnx
mkdir onnx
python3.7 pth2onnx.py pth/siamfc.pth onnx/exemplar.onnx onnx/search.onnx
# set environment
source env.sh
# generate om
rm -rf om
atc --model=./onnx/exemplar.onnx --framework=5 --output=./om/exemplar_bs1 --input_format=NCHW --input_shape="actual_input_1:1,3,127,127" --log=debug --soc_version=Ascend310
atc --model=./onnx/search.onnx --framework=5 --output=./om/search_bs1 --input_format=NCHW --input_shape="actual_input_1:1,9,255,255" --log=debug --soc_version=Ascend310
if [ -f "om/exemplar_bs1.om" ] && [ -f "om/search_bs1.om" ]; then
    echo "success"
else
    echo "fail!"
fi