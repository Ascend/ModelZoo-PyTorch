#!/bin/bash

set -eu
batch_size=1

for para in $*
do
    if [[ $para == --batch_size* ]]; then
        batch_size=`echo ${para#*=}`
    fi
done


cd mmdetection
rm -f ../yolox.onnx

python tools/deployment/pytorch2onnx.py configs/yolox/yolox_x_8x8_300e_coco.py ../yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth --output-file ../yolox.onnx --shape 640 640 --dynamic-export

if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

cd ..
rm -f yolox.om
source env.sh

atc --framework=5 --model=yolox.onnx --output=yolox  --input_format=NCHW --input_shape="input:$batch_size,3,640,640" --log=error --soc_version=Ascend710

if [ -f "yolox.om" ]; then
    echo "success"
else
    echo "fail!"
fi
