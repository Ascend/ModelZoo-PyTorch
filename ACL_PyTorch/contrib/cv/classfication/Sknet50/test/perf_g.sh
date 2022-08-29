#!/bin/bash

echo "=========bs1========="
trtexec --onnx=sk_resnet50.onnx --fp16 --shapes=image:1x3x224x224 --threads
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

echo "=========bs64========="
trtexec --onnx=sk_resnet50.onnx --fp16 --shapes=image:64x3x224x224 --threads
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi