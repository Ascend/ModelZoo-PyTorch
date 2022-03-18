#!/usr/bin/env bash

rm -rf m2det512.onnx
python3.7 M2Det_pth2onnx.py -c=M2Det/configs/m2det512_vgg.py -pth=M2Det/weights/m2det512_vgg.pth -onnx=m2det512.onnx

source env.sh
rm -rf m2det512_bs1.om
atc --framework=5 --model=m2det512.onnx --input_format=NCHW --input_shape="image:1,3,512,512" --output=m2det512_bs1 --log=debug --soc_version=Ascend310 --out-nodes="Softmax_1234:0;Reshape_1231:0"
rm -rf m2det512_bs16.om
atc --framework=5 --model=m2det512.onnx --input_format=NCHW --input_shape="image:16,3,512,512" --output=m2det512_bs16 --log=debug --soc_version=Ascend310 --out-nodes="Softmax_1234:0;Reshape_1231:0"
if [ -f "m2det512_bs1.om" ] && [ -f "m2det512_bs16.om" ]; then
    echo "success"
else
    echo "fail!"
fi

