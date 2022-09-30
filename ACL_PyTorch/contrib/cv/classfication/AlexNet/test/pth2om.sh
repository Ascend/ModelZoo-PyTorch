#!/bin/bash

# 生成onnx文件
rm -rf alexnet.onnx
python3.7 pth2onnx.py alexnet-owt-4df8aa71.pth alexnet.onnx
# 配置环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# 生成om文件
rm -rf onnx_alexnet_bs1.om onnx_alexnet_bs16.om
atc --model=./alexnet.onnx --framework=5 --output=./onnx_alexnet_bs1 --input_format=NCHW --input_shape="actual_input_1:1,3,224,224" --log=debug --soc_version=Ascend310
atc --model=./alexnet.onnx --framework=5 --output=./onnx_alexnet_bs16 --input_format=NCHW --input_shape="actual_input_1:16,3,224,224" --log=debug --soc_version=Ascend310
if [ -f "onnx_alexnet_bs1.om" ] && [ -f "onnx_alexnet_bs16.om" ]; then
    echo "success"
else
    echo "fail!"
fi