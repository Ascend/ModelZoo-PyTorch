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

source env.sh

export install_path=/usr/local/Ascend/ascend-toolkit/latest
export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export PYTHONPATH=${install_path}/atc/python/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=${install_path}/atc/lib64:${install_path}/acllib/lib64:$LD_LIBRARY_PATH
export ASCEND_OPP_PATH=${install_path}/opp
export ASCEND_AICPU_PATH=/usr/local/Ascend/ascend-toolkit/latest

atc --framework=5 --model=TextCNN.onnx --output=TextCNN_bs1 --input_format=ND --input_shape="sentence:1,32" --log=error --soc_version=Ascend310

atc --framework=5 --model=TextCNN.onnx --output=TextCNN_bs16 --input_format=ND --input_shape="sentence:16,32" --log=error --soc_version=Ascend310

if [ -f "TextCNN_bs1.om" ] && [ -f "TextCNN_bs16.om" ]; then
    echo "success"
else
    echo "fail!"
fi

