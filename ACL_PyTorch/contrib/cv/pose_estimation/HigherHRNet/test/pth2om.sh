#!/bin/bash
set -eu
mkdir -p models
rm -rf models/*.onnx
python pth2onnx.py --cfg ./HigherHRNet-Human-Pose-Estimation/experiments/coco/higher_hrnet/w32_512_adam_lr1e-3.yaml --weights models/pose_higher_hrnet_w32_512.pth

if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

rm -rf models/*.om
source env.sh
#--out_nodes="Conv_1818:0;Conv_1851:0"
#--out_nodes="Conv_770:0;Conv_795:0"
atc --framework=5 \
    --model=models/pose_higher_hrnet_w32_512_bs1_dynamic.onnx \
    -output=models/pose_higher_hrnet_w32_512_bs1_dynamic --input_format=NCHW \
    --input_shape="input:1,3,-1,-1" \
    --dynamic_image_size="1024,512;960,512;896,512;832,512;768,512;704,512;640,512;576,512;512,512;512,576;512,640;512,704;512,768;512,832;512,896;512,960;512,1024" \
    --out_nodes="Conv_770:0;Conv_795:0"\
    --log=debug  \
    --soc_version=Ascend310


if  [ -f "models/pose_higher_hrnet_w32_512_bs1.om" ]; then
    echo "success"
else
    echo "fail!"
fi