#!/bin/bash

rm -rf DnCNN-S-15.onnx
python3.7 DnCNN_pth2onnx.py net.pth DnCNN-S-15.onnx

source env.sh

rm -rf DnCNN-S-15_bs1.om DnCNN-S-15_bs16.om
atc --framework=5 --model=DnCNN-S-15.onnx --input_format=NCHW --input_shape="actual_input_1:1,1,481,481" --output=DnCNN-S-15_bs1 --log=debug --soc_version=Ascend310
atc --framework=5 --model=DnCNN-S-15.onnx --input_format=NCHW --input_shape="actual_input_1:16,1,481,481" --output=DnCNN-S-15_bs16 --log=debug --soc_version=Ascend310

if [ -f "DnCNN-S-15_bs1.om" ] && [ -f "DnCNN-S-15_bs16.om" ]; then
    echo "success"
else
    echo "fail!"
fi
