#!/bin/bash

rm -rf mobilenet-v1.onnx
python3.7 mobilenet-v1_pth2onnx.py mobilenet_sgd_rmsprop_69.526.tar mobilenet-v1.onnx
source env.sh
rm -rf mobilenet-v1_bs1.om mobilenet-v1_bs16.om
atc --framework=5 --model=./mobilenet-v1.onnx --input_format=NCHW --input_shape="image:1,3,224,224" --output=mobilenet-v1_bs1 --log=debug --soc_version=Ascend310
atc --framework=5 --model=./mobilenet-v1.onnx --input_format=NCHW --input_shape="image:16,3,224,224" --output=mobilenet-v1_bs16 --log=debug --soc_version=Ascend310
if [ -f "mobilenet-v1_bs1.om" ] && [ -f "mobilenet-v1_bs16.om" ]; then
    echo "success"
else
    echo "fail!"
fi
