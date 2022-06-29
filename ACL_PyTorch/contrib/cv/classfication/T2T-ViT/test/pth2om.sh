#!/bin/bash

# 生成onnx文件
python3.7 pth2onnx.py

# 配置环境变量
source env.sh

# 生成om文件
atc --framework=5 --model=T2T_ViT_14.onnx --output=T2T_ViT_14_bs8_test --input_format=NCHW --input_shape="input:8,3,224,224" --soc_version=Ascend710 --keep_dtype=keep_dtype.cfg