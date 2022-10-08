#!/bin/bash

rm -rf mb1-ssd.onnx
python3.7 SSD_MobileNet_pth2onnx.py mobilenet-v1-ssd.pth mb1-ssd.onnx
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
rm -rf mb1-ssd_fix.onnx
python3 fix_softmax_transpose.py mb1-ssd.onnx mb1-ssd_fix.onnx

rm -rf mb1-ssd_bs1.om mb1-ssd_bs16.om
source /usr/local/Ascend/ascend-toolkit/set_env.sh
atc --framework=5 --model=mb1-ssd_fix.onnx --output=mb1-ssd_bs1 --input_format=NCHW --input_shape="image:1,3,300,300" --log=debug --soc_version=Ascend310
atc --framework=5 --model=mb1-ssd_fix.onnx --output=mb1-ssd_bs16 --input_format=NCHW --input_shape="image:16,3,300,300" --log=debug --soc_version=Ascend310

if [ -f "mb1-ssd_bs1.om" ] && [ -f "mb1-ssd_bs16.om" ]; then
    echo "success"
else
    echo "fail!"
fi