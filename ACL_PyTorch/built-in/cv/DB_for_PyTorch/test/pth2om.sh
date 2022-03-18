#!/bin/bash

cd DB
rm -rf dbnet.onnx
python3.7 ../db_pth2onnx.py experiments/seg_detector/ic15_resnet50_deform_thre.yaml --resume ./ic15_resnet50
source ../env.sh
rm -rf db_bs1.om db_bs16.om
atc --framework=5 --model=./dbnet.onnx --input_format=NCHW --input_shape="actual_input_1:1,3,736,1280" --output=db_bs1 --log=debug --soc_version=Ascend310
atc --framework=5 --model=./dbnet.onnx --input_format=NCHW --input_shape="actual_input_1:16,3,736,1280" --output=db_bs16 --log=debug --soc_version=Ascend310
if [ -f "db_bs1.om" ] && [ -f "db_bs16.om" ]; then
    echo "success"
else
    echo "fail!"
fi
