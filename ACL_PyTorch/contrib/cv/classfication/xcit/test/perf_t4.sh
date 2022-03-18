#!/bin/bash

source env.sh

python3.7 xcit_pth2onnx.py --pretrained=pretrained/xcit_small_12_p16_224.pth --fp16
python3.7 xcit_pth2onnx.py --pretrained=pretrained/xcit_small_12_p16_224.pth --batch-size=16 --fp16
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
python onnx_test.py speed onnx_models/xcit_b1_fp16.onnx 1
python onnx_test.py speed onnx_models/xcit_b16_fp16.onnx 16