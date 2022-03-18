#!/bin/bash

source env.sh

echo 'pth -> onnx'
rm -rf PointRend.onnx
python3.7 PointRend_pth2onnx.py 'detectron2/projects/PointRend/configs/SemanticSegmentation/pointrend_semantic_R_101_FPN_1x_cityscapes.yaml' './model_final_cf6ac1.pkl' './PointRend.onnx'
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

rm -rf PointRend_bs1.om
echo 'onnx -> om batch1'
atc --framework=5 --model=./PointRend.onnx --output=PointRend_bs1 --input_format=NCHW --input_shape="images:1,3,1024,2048" --log=debug --soc_version=Ascend310
if [ -f "PointRend_bs1.om" ]; then
    echo "success"
else
    echo "fail!"
fi