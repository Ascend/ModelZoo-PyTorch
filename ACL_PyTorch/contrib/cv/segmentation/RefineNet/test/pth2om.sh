#!/bin/bash

rm -rf model/RefineNet_910.onnx
python RefineNet_pth2onnx.py --input-file model/RefineNet_910.pth.tar --output-file model/RefineNet_910.onnx
source /usr/local/Ascend/ascend-toolkit/set_env.sh
rm -rf model/RefineNet_910_bs1.om model/RefineNet_910_bs16.om
atc --framework=5 --model=model/RefineNet_910.onnx --output=model/RefineNet_910_bs1 --input_format=NCHW --input_shape="input:1,3,500,500" --log=debug --soc_version=Ascend310
atc --framework=5 --model=model/RefineNet_910.onnx --output=model/RefineNet_910_bs16 --input_format=NCHW --input_shape="input:16,3,500,500" --log=debug --soc_version=Ascend310
if [ -f "model/RefineNet_910_bs1.om" ] && [ -f "model/RefineNet_910_bs16.om" ]; then
    echo "success"
else
    echo "fail!"
fi