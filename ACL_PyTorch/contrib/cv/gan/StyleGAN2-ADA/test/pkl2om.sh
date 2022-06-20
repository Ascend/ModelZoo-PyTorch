#!/bin/bash

if [ -f "./*.onnx" ]; then
  rm -f `ls *.onnx`
fi

python pkl2onnx.py --batch_size=1
python pkl2onnx.py --batch_size=16

if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

if [ -f "./*.om"]; then
  rm -f `ls *.om`
fi

source env.sh

atc --framework=5 --model=./G_ema_onnx_bs1.onnx --output=G_ema_om_bs1 --input_format=ND --input_shape="z:1,512" --log=debug --soc_version=Ascend310 --buffer_optimize=off_optimize
atc --framework=5 --model=./G_ema_onnx_bs16.onnx --output=G_ema_om_bs16 --input_format=ND --input_shape="z:16,512" --log=debug --soc_version=Ascend310 --buffer_optimize=off_optimize

if [ -f "G_ema_om_bs1.om" ] && [ -f "G_ema_om_bs16.om" ]; then
    echo "success!"
else
    echo "fail!"
fi