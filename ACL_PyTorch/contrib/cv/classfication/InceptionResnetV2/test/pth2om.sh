#!/bin/bash

rm -rf inceptionresnetv2.onnx
python3.7 inceptionresnetv2_pth2onnx.py inceptionresnetv2-520b38e4.pth inceptionresnetv2.onnx
source env.sh
rm -rf inceptionresnetv2-b0_bs1.om inceptionresnetv2-b0_bs16.om
atc --framework=5 --model=inceptionresnetv2.onnx --output=inceptionresnetv2-b0_bs1 --input_format=NCHW --input_shape="image:1,3,299,299" --log=debug --soc_version=Ascend310
atc --framework=5 --model=inceptionresnetv2.onnx --output=inceptionresnetv2-b0_bs16 --input_format=NCHW --input_shape="image:16,3,299,299" --log=debug --soc_version=Ascend310
if [ -f "inceptionresnetv2-b0_bs1.om" ] && [ -f "inceptionresnetv2-b0_bs16.om" ]; then
    echo "success"
else
    echo "fail!"
fi
