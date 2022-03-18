#!/bin/bash

if [ -f "srgan.onnx" ]; then
	rm srgan.onnx
fi

python3.7 srgan_pth2onnx.py netG_best.pth

if [ $? == 0 ]; then
    echo "Convert onnx success!"
else
    echo "fail!"
    exit -1
fi

python3.7 eidt_onnx.py
if [ $? == 0 ]; then
    echo "Edit onnx success!"
else
    echo "fail!"
    exit -1
fi



rm -rf srgan_dynamic.om

source env.sh
atc --model=./srgan_fix.onnx --framework=5 --output=srgan_dynamic --input_format=NCHW --input_shape="lrImage:1,3,-1,-1" --dynamic_image_size="140,140;256,256;172,114;128,128;144,144" --log=info --soc_version=Ascend310

if [ -f "srgan_dynamic.om" ]; then
    echo "Convert om success!"
else
    echo "fail!"
fi
