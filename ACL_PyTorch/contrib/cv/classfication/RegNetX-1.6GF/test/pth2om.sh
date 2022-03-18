#!/bin/bash

rm -rf RegNetX-1.6GF.onnx
python3.7 RegNetX_onnx.py regnetx_016-65ca972a.pth RegNetX-1.6GF.onnx
source env.sh
rm -rf RegNetX-1.6GF_bs1.om RegNetX-1.6GF.om
atc --framework=5 --model=./RegNetX-1.6GF.onnx --input_format=NCHW --input_shape="image:1,3,224,224" --output=RegNetX-1.6GF_bs1 --log=debug --soc_version=Ascend310
atc --framework=5 --model=./RegNetX-1.6GF.onnx --input_format=NCHW --input_shape="image:16,3,224,224" --output=RegNetX-1.6GF_bs16 --log=debug --soc_version=Ascend310
if [ -f "RegNetX-1.6GF_bs1.om" ] && [ -f "RegNetX-1.6GF_bs16.om" ]; then
    echo "success"
else
    echo "fail!"
fi