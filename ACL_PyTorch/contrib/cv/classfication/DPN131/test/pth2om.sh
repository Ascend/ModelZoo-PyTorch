#!/bin/bash
rm -rf dpn131.onnx
pip3.7 uninstall pretrainedmodels
python3.7 dpn131_pth2onnx.py dpn131-7af84be88.pth dpn131.onnx

source /usr/local/Ascend/ascend-toolkit/set_env.sh

rm -rf dpn131_bs1.om dpn131_bs16.om
atc --framework=5 --model=./dpn131.onnx --input_format=NCHW --input_shape="image:1,3,224,224" --output=dpn131_bs1 --log=debug --soc_version=Ascend310
atc --framework=5 --model=./dpn131.onnx --input_format=NCHW --input_shape="image:16,3,224,224" --output=dpn131_bs16 --log=debug --soc_version=Ascend310
if [ -f "dpn131_bs1.om" ] && [ -f "dpn131_bs16.om" ]; then
    echo "success"
else
    echo "fail!"
fi