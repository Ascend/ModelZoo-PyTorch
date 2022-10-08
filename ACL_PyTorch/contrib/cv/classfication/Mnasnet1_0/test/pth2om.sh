#!/bin/bash

rm -rf mnasnet1.0.onnx
python3.7 mnasnet_pth2onnx.py ./mnasnet1.0_top1_73.512-f206786ef8.pth mnasnet1.0.onnx
source /usr/local/Ascend/ascend-toolkit/set_env.sh
rm -rf mnasnet1.0_bs1.om mnasnet1.0_bs16.om

atc --framework=5 --model=./mnasnet1.0.onnx --input_format=NCHW --input_shape="image:1,3,224,224" --output=mnasnet1.0_bs1 --log=debug --soc_version=Ascend310
atc --framework=5 --model=./mnasnet1.0.onnx --input_format=NCHW --input_shape="image:16,3,224,224" --output=mnasnet1.0_bs16 --log=debug --soc_version=Ascend310
if [ -f "mnasnet1.0_bs1.om" ] && [ -f "mnasnet1.0_bs16.om" ]; then
    echo "success"
else
    echo "fail!"
fi
