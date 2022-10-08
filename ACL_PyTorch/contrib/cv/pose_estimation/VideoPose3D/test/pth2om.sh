#!/bin/bash

model_path='./VideoPose3D/checkpoint/model_best.bin'

for para in $*
do
    if [[ $para == --model_path* ]]; then
        model_path=`echo ${para#*=}`
    fi
done

rm -rf vp3d.onnx
python vp3d_pth2onnx.py -m ${model_path} -o vp3d.onnx

if [ $? != 0 ]; then
    echo "pth to onnx fail!"
    exit -1
fi

rm -rf vp3d_seq6115.om
source /usr/local/Ascend/ascend-toolkit/set_env.sh
atc --framework=5 --model=vp3d.onnx --output=vp3d_seq6115 --input_format=NCHW --input_shape="2d_poses:2,6115,17,2" --log=debug --soc_version=Ascend310

if [ -f "vp3d_seq6115.om" ]; then
    echo "onnx to om success"
else
    echo "onnx to om fail!"
    exit -1
fi
