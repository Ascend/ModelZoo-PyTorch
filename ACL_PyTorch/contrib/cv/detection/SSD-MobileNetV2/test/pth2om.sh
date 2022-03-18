#!/bin/bash

rm -rf SSD-MobileNetV2.onnx
python3.7 SSD-MobileNetV2_pth2onnx.py mb2-ssd-lite-mp-0_686.pth base_net SSD-MobileNetV2.onnx
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

rm -rf SSD-MobileNetV2_adapt.onnx
python3.7 SSD-MobileNetV2_adapt.py SSD-MobileNetV2.onnx SSD-MobileNetV2_adapt.onnx
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
rm -rf SSD-MobileNetV2_bs1.om SSD-MobileNetV2_bs16.om
source env.sh
atc --framework=5 --model=SSD-MobileNetV2_adapt.onnx --output=SSD-MobileNetV2_bs1 --input_format=NCHW --input_shape="image:1,3,300,300" --auto_tune_mode="RL,GA" --log=debug --soc_version=Ascend310
atc --framework=5 --model=SSD-MobileNetV2_adapt.onnx --output=SSD-MobileNetV2_bs16 --input_format=NCHW --input_shape="image:16,3,300,300" --auto_tune_mode="RL,GA" --log=debug --soc_version=Ascend310

if [ -f "SSD-MobileNetV2_bs1.om" ] && [ -f "SSD-MobileNetV2_bs16.om" ]; then
    echo "success"
else
    echo "fail!"
fi