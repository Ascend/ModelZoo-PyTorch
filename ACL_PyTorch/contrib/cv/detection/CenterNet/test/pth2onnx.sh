#!/bin/bash

rm -rf CenterNet.onnx
python3.7 CenterNet_pth2onnx.py ctdet_coco_dla_2x.pth CenterNet.onnx
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi