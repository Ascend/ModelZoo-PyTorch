#!/bin/bash

python -m tf2onnx.convert --input resnet50_v1.pb --inputs input_tensor:0 --rename-inputs dummy_input --inputs-as-nchw dummy_input --outputs ArgMax:0 --output resnet50_tmp.onnx --opset 11

python re_domain.py resnet50_tmp.onnx resnet50.onnx

rm -rf resnet50_tmp.onnx
