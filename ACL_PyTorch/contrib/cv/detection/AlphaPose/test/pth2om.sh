#!/bin/bash

set -eu

mkdir -p models
rm -rf models/*.onnx
python3.7 pth2onnx.py AlphaPose/configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml models/fast_res50_256x192.pth models/fast_res50_256x192.onnx
python3.7 -m onnxsim --input-shape="1,3,256,192" models/fast_res50_256x192.onnx models/fast_res50_256x192_bs1.onnx
python3.7 -m onnxsim --input-shape="16,3,256,192" models/fast_res50_256x192.onnx models/fast_res50_256x192_bs16.onnx

if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

rm -rf models/*.om
source /usr/local/Ascend/ascend-toolkit/set_env.sh

atc --framework=5 --model=models/fast_res50_256x192_bs1.onnx --output=models/fast_res50_256x192_bs1 --input_format=NCHW --input_shape="image:1,3,256,192" --log=debug --soc_version=Ascend310
atc --framework=5 --model=models/fast_res50_256x192_bs16.onnx --output=models/fast_res50_256x192_bs16 --input_format=NCHW --input_shape="image:16,3,256,192" --log=debug --soc_version=Ascend310


if [ -f "models/fast_res50_256x192_bs1.om" ] && [ -f "models/fast_res50_256x192_bs16.om" ]; then
    echo "success"
else
    echo "fail!"
fi
