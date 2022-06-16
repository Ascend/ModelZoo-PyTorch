#!/bin/bash

install_path=$1
source ${install_path}/ascend-toolkit/set_env.sh

echo 'pth -> onnx'
rm -rf ./se_resnet50_dynamic_bs.onnx
python3 SE_ResNet50_pth2onnx.py ./se_resnet50-ce0d4300.pth ./se_resnet50_dynamic_bs.onnx
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

echo 'onnx -> om batch32'
rm -rf ./se_resnet50_bs32.om
atc --model=./se_resnet50_dynamic_bs.onnx --framework=5 --input_format=NCHW --input_shape="image:32,3,224,224" --output=./se_resnet50_fp16_bs32 --log=error --soc_version=$2 --insert_op_conf=./aipp_SE_ResNet50_pth.config --enable_small_channel=1
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

