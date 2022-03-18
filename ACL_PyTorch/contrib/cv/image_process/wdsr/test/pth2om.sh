#!/bin/bash

rm -rf wdsr.onnx
python3.7 Wdsr_pth2onnx.py --ckpt epoch_30.pth --model wdsr --output_name wdsr.onnx --scale 2

source env.sh
rm -rf Wdsr_bs1.om Wdsr_bs8.om
atc --framework=5 --model=wdsr.onnx --output=wdsr_bs1 --input_format=NCHW --input_shape="image:1,3,1020,1020" --log=debug --soc_version=Ascend310
atc --framework=5 --model=wdsr.onnx --output=wdsr_bs8 --input_format=NCHW --input_shape="image:8,3,1020,1020" --log=debug --soc_version=Ascend310
if [ -f "wdsr_bs1.om" ] && [ -f "wdsr_bs8.om" ]; then
    echo "success"
else
    echo "fail!"
fi

