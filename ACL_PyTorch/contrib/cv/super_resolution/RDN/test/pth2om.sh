#!/bin/bash

rm -rf rdn_x2.onnx
python3.7 RDN_pth2onnx.py --input-file=rdn_x2.pth --output-file=rdn_x2.onnx --num-features=64 --growth-rate=64 --num-blocks=16 --num-layers=8 --scale=2
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

rm -rf rdn_x2_bs1.om
source /usr/local/Ascend/ascend-toolkit/set_env.sh
atc --framework=5 --model=rdn_x2.onnx --output=rdn_x2_bs1 --input_format=NCHW --input_shape="image:1,3,114,114" --log=debug --soc_version=Ascend310P

if [ -f "rdn_x2_bs1.om" ]; then
    echo "success"
else
    echo "fail!"
fi
