#!/bin/bash
export workdir=`pwd`/mmdetection/
echo 'pth -> onnx'
rm -rf GCNet.onnx
python $workdir/tools/deployment/pytorch2onnx.py $workdir/configs/gcnet/mask_rcnn_r50_fpn_r4_gcb_c3-c5_1x_coco.py $workdir/mask_rcnn_r50_fpn_r4_gcb_c3-c5_1x_coco_20200204-17235656.pth --output-file $workdir/GCNet.onnx --input-img $workdir/demo/demo.jpg --test-img $workdir/tests/data/color.jpg --shape 800 1216 
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo 'onnx -> om batch1'
rm -rf GCNet_bs1.om
atc --framework=5 --model=$workdir/GCNet.onnx --output=./GCNet_bs1 --input_shape="input:1,3,800,1216"  --log=error --soc_version=Ascend310
if [ -f "GCNet_bs1.om" ]; then
    echo "success"
else
    echo "fail!"
fi
