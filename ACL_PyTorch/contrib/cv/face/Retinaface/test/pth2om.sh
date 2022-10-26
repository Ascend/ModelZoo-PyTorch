#!/bin/bash

currentDir=$(cd "$(dirname "$0")";pwd)/..
rm -rf retinaface.onnx
python3.7 pth2onnx.py -m $currentDir/Pytorch_Retinaface/weights/mobilenet0.25_Final.pth
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
source /usr/local/Ascend/ascend-toolkit/set_env.sh
rm -rf retinaface_bs1.om retinaface_bs16.om
atc --framework 5 --model retinaface.onnx --input_shape "image:1,3,1000,1000" --soc_version $1 --output retinaface_bs1 --log error --out-nodes="Concat_205:0;Softmax_206:0;Concat_155:0" --enable_small_channel=1 --insert_op_conf=./aipp.cfg
atc --framework 5 --model retinaface.onnx --input_shape "image:16,3,1000,1000" --soc_version $1 --output retinaface_bs16 --log error --out-nodes="Concat_205:0;Softmax_206:0;Concat_155:0" --enable_small_channel=1 --insert_op_conf=./aipp.cfg

if [ -f "retinaface_bs1.om" ] && [ -f "retinaface_bs16.om" ]; then
    echo "success"
else
    echo "fail!"
fi
