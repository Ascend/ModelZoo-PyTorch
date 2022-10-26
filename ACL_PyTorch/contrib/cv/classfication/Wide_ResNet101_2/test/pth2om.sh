#!/bin/bash
rm -rf wrn101_2_pth.onnx
python3.7 wrn101_2_pth2onnx.py wide_resnet101_2-32ee1156.pth wrn101_2_pth.onnx
source /usr/local/Ascend/ascend-toolkit/set_env.sh
rm -rf wrn101_2_bs1.om wrn101_2_bs16.om
atc --framework=5 --model=./wrn101_2_pth.onnx --input_format=NCHW --input_shape="image:1,3,224,224" --output=wrn101_2_bs1 --log=debug --soc_version=Ascend310
atc --framework=5 --model=./wrn101_2_pth.onnx --input_format=NCHW --input_shape="image:16,3,224,224" --output=wrn101_2_bs16 --log=debug --soc_version=Ascend310
if [ -f "wrn101_2_bs1.om" ] && [ -f "wrn101_2_bs16.om" ]; then
    echo "success"
else
    echo "fail!"
fi