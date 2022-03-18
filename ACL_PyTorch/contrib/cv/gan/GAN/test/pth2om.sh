#!/bin/bash

rm -rf GAN.onnx
python3.7 GAN_pth2onnx.py --input_file=generator_8p_0.0008_128.pth --output_file=GAN.onnx
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

rm -rf GAN_bs1.om
source env.sh
export REPEAT_TUNE=True
atc --model=GAN.onnx --framework=5 --output=GAN_bs1 --input_format=NCHW --input_shape="Z:1,100" --log=error --soc_version=Ascend310 --auto_tune_mode="RL,GA"

if [ -f "GAN_bs1.om" ]; then
    echo "Turn to GAN_bs1.om success!"
else
    echo "Turn to GAN_bs1.om fail!"
fi

rm -rf GAN_bs16.om
atc --model=GAN.onnx --framework=5 --output=GAN_bs16 --input_format=NCHW --input_shape="Z:16,100" --log=error --soc_version=Ascend310 --auto_tune_mode="RL,GA"

if [ -f "GAN_bs16.om" ]; then
    echo "Turn to GAN_bs16.om success!"
else
    echo "Turn to GAN_bs16.om fail!"
fi

rm -rf GAN_bs64.om
atc --model=GAN.onnx --framework=5 --output=GAN_bs64 --input_format=NCHW --input_shape="Z:64,100" --log=error --soc_version=Ascend310

if [ -f "GAN_bs64.om" ]; then
    echo "Turn to GAN_bs64.om success!"
else
    echo "Turn to GAN_bs64.om fail!"
fi