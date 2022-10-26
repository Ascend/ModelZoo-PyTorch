#!/bin/bash

rm -rf intraDA_deeplabv2.onnx
python3.7 intrada_pth2onnx.py ./cityscapes_easy2hard_intrada_with_norm.pth ./intraDA_deeplabv2.onnx
source /usr/local/Ascend/ascend-toolkit/set_env.sh
rm -rf intraDA_deeplabv2_bs1.om intraDA_deeplabv2_bs16.om
atc --framework=5 --model=./intraDA_deeplabv2.onnx --output=intraDA_deeplabv2_bs1 --input_format=NCHW --input_shape="image:1,3,512,1024" --log=debug --soc_version=Ascend${chip_name}
atc --framework=5 --model=./intraDA_deeplabv2.onnx --output=intraDA_deeplabv2_bs16 --input_format=NCHW --input_shape="image:16,3,512,1024" --log=debug --soc_version=${chip_name}
if [ -f "intraDA_deeplabv2_bs1.om" ] && [ -f "intraDA_deeplabv2_bs16.om" ]; then
    echo "success"
else
    echo "fail!"
fi