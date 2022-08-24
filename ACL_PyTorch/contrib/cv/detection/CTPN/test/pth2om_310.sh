#!/bin/bash

rm -rf ctpn.onnx
python3.7 ctpn_pth2onnx.py --pth_path=./ctpn.pytorch/weights/ctpn.pth --onnx_path=ctpn.onnx 
if [ $? != 0 ]; then
    echo "pth to onnx fail!"
    exit -1
fi

python3.7 task_process.py --mode='change model'
if [ $? != 0 ]; then
    echo "change onnx model fail!"
    exit -1
fi

rm -rf ctpn_bs1_310.om
source env.sh
atc --framework=5 --model=ctpn_change_1000x462.onnx --output=ctpn_bs1_310 --input_format=NCHW --input_shape="image:1,3,-1,-1" --dynamic_image_size="248,360;280,550;319,973;458,440;477,636;631,471;650,997;753,1000;997,744;1000,462" --log=debug --soc_version=Ascend310

if [ -f "ctpn_bs1_310.om" ]; then
    echo "onnx to om success"
else
    echo "onnx to om fail!"
    exit -1
fi