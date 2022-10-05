#!/bin/bash

rm -rf TextCNN.onnx
python3.7 TextCNN_pth2onnx.py \
--weight_path ./Chinese-Text-Classification-Pytorch/THUCNews/saved_dict/TextCNN_9045_seed460473.pth \
--onnx_path TextCNN.onnx

if [ $? != 0 ]; then
	echo "fail!"
	exit -1
fi

rm -rf TextCNN_bs1.om TextCNN_bs16.om

# 该脚本中环境变量仅供参考，请以实际安装环境配置比环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh

atc --framework=5 --model=TextCNN.onnx --output=TextCNN_bs1 --input_format=ND --input_shape="sentence:1,32" --log=error --soc_version=Ascend310

atc --framework=5 --model=TextCNN.onnx --output=TextCNN_bs16 --input_format=ND --input_shape="sentence:16,32" --log=error --soc_version=Ascend310

if [ -f "TextCNN_bs1.om" ] && [ -f "TextCNN_bs16.om" ]; then
    echo "success"
else
    echo "fail!"
fi

