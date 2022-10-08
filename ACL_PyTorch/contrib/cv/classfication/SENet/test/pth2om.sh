#!/bin/bash

rm -rf se_resnet50.onnx
pip3.7 uninstall pretrainedmodels
python3.7 se_resnet50_pth2onnx.py se_resnet50-ce0d4300.pth se_resnet50.onnx
source /usr/local/Ascend/ascend-toolkit/set_env.sh
rm -rf se_resnet50_bs1.om se_resnet50_bs16.om
atc --framework=5 --model=./se_resnet50.onnx --input_format=NCHW --input_shape="image:1,3,224,224" --output=se_resnet50_bs1 --log=debug --soc_version=Ascend310
atc --framework=5 --model=./se_resnet50.onnx --input_format=NCHW --input_shape="image:16,3,224,224" --output=se_resnet50_bs16 --log=debug --soc_version=Ascend310
if [ -f "se_resnet50_bs1.om" ] && [ -f "se_resnet50_bs16.om" ]; then
    echo "success"
else
    echo "fail!"
fi