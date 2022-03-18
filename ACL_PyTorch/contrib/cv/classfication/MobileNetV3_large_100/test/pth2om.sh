#!/bin/bash

rm -rf mobilenetv3_100.onnx
python3.7 MobileNetV3_pth2onnx.py mobilenetv3_large_100_ra-f55367f5.pth mobilenetv3_100.onnx
source env.sh
rm -rf mobilenetv3_100_bs1.om mobilenetv3_100_bs16.om
atc --framework=5 --model=./mobilenetv3_100.onnx --input_format=NCHW --input_shape="image:1,3,224,224" --output=mobilenetv3_100_bs1 --log=debug --soc_version=Ascend310
atc --framework=5 --model=./mobilenetv3_100.onnx --input_format=NCHW --input_shape="image:16,3,224,224" --output=mobilenetv3_100_bs16 --log=debug --soc_version=Ascend310
if [ -f "mobilenetv3_100_bs1.om" ] && [ -f "mobilenetv3_100_bs16.om" ]; then
    echo "success"
else
    echo "fail!"
fi
