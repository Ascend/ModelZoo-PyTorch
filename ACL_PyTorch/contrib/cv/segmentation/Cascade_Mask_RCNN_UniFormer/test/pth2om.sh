#!/bin/bash

set -eu
height=800
width=1216

rm -f uniformer_bs1.onnx
cd UniFormer/object_detection

python tools/deployment/pytorch2onnx.py \
    exp/cascade_mask_rcnn_3x_ms_hybrid_base/config.py \
    ../../cascade_mask_rcnn_3x_ms_hybrid_base.pth \
    --input-img demo/demo.jpg \
    --output-file ../../uniformer_bs1.onnx \
    --shape $height $width

if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

cd ../..
rm -f uniformer_bs1.om
source /usr/local/Ascend/ascend-toolkit/set_env.sh

atc --framework=5 --model=uniformer_bs1.onnx --output=uniformer_bs1 --input_format=NCHW \
    --input_shape="input:1,3,$height,$width" --log=error --soc_version=Ascend710

if [ -f "uniformer_bs1.om" ]; then
    echo "success"
else
    echo "fail!"
fi
