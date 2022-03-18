#!/bin/bash

rm -rf RegNetY-1.6GF.onnx
python3.7 RegNetY_onnx.py regnety_016-54367f74.pth RegNetY-1.6GF.onnx
source env.sh
rm -rf RegNetY-1.6GF_bs1.om RegNetY-1.6GF.om
atc --framework=5 --model=./RegNetY-1.6GF.onnx --input_format=NCHW --input_shape="image:1,3,224,224" --output=RegNetY-1.6GF_bs1 --log=debug --soc_version=Ascend310
atc --framework=5 --model=./RegNetY-1.6GF.onnx --input_format=NCHW --input_shape="image:16,3,224,224" --output=RegNetY-1.6GF_bs16 --log=debug --soc_version=Ascend310
if [ -f "RegNetY-1.6GF_bs1.om" ] && [ -f "RegNetY-1.6GF_bs16.om" ]; then
    echo "success"
else
    echo "fail!"
fi