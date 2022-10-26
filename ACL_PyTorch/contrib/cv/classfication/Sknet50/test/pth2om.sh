#! /bin/bash

rm -rf sk_resnet50.onnx
rm -rf sk_resnet50_bs1_310P.om
rm -rf sk_resnet50_bs8_310P.om   #batch8最优
rm -rf fusion_result.json

python sknet2onnx.py --pth sk_resnet50.pth.tar --onnx sk_resnet50
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
source /usr/local/Ascend/ascend-toolkit/set_env.sh
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
atc --framework=5 --model=sk_resnet50.onnx --output=sk_resnet50_bs1_310P --input_format=NCHW --input_shape="image:1,3,224,224" --log=debug --soc_version=$1
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
atc --framework=5 --model=sk_resnet50.onnx --output=sk_resnet50_bs8_310P --input_format=NCHW --input_shape="image:8,3,224,224" --log=debug --soc_version=$1
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
if [ -f "sk_resnet50_bs1_310P.om" ] && [ -f "sk_resnet50_bs8_310P.om" ]; then
    echo "success"
else
    echo "fail!"
fi