#!/bin/bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh

rm -rf RepVGG.onnx
python RepVGG_pth2onnx.py RepVGG-A0-train.pth RepVGG.onnx
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

rm -rf RepVGG_bs1.om RepVGG_bs16.om
atc --framework=5 --model=RepVGG.onnx --output=RepVGG_bs1 --input_format=NCHW --input_shape="actual_input_1:1,3,224,224" --log=error --soc_version=${1}
atc --framework=5 --model=RepVGG.onnx --output=RepVGG_bs16 --input_format=NCHW --input_shape="actual_input_1:16,3,224,224" --log=error --soc_version=${1}

if [ -f "RepVGG_bs1.om" ] && [ -f "RepVGG_bs16.om" ]; then
    echo "success"
else
    echo "fail!"
fi
