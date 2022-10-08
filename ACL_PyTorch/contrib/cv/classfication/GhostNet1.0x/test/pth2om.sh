#!/bin/bash

rm -rf ghostnet.onnx
python3.7.5 ghostnet_pth2onnx.py state_dict_73.98.pth ghostnet.onnx
source /usr/local/Ascend/ascend-toolkit/set_env.sh
rm -rf ghostnet_bs1.om ghostnet_bs16.om
atc --framework=5 --model=./ghostnet.onnx --input_format=NCHW --input_shape="image:1,3,224,224" --output=ghostnet_bs1 --log=debug --soc_version=Ascend310P3
atc --framework=5 --model=./ghostnet.onnx --input_format=NCHW --input_shape="image:16,3,224,224" --output=ghostnet_bs16 --log=debug --soc_version=Ascend310P3
if [ -f "ghostnet_bs1.om" ] && [ -f "ghostnet_bs16.om" ]; then
    echo "success"
else
    echo "fail!"
fi
