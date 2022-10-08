#!/bin/bash

rm -rf fcos.onnx
python3.7 mmdetection/tools/deployment/pytorch2onnx.py mmdetection/configs/fcos/fcos_r50_caffe_fpn_4x4_1x_coco.py ./fcos_r50_caffe_fpn_1x_4gpu_20190516-a7cac5ff.pth --output-file fcos.onnx --input-img mmdetection/demo/demo.jpg --test-img mmdetection/tests/data/color.jpg --shape 800 1333
source /usr/local/Ascend/ascend-toolkit/set_env.sh
rm -rf fcos_bs1.om fcos_bs16.om
atc --framework=5 --model=./fcos.onnx --output=fcos_bs1 --input_format=NCHW --input_shape="input:1,3,800,1333" --log=debug --soc_version=Ascend310 --out_nodes="labels;dets"
atc --framework=5 --model=./fcos.onnx --output=fcos_bs16 --input_format=NCHW --input_shape="input:16,3,800,1333" --log=debug --soc_version=Ascend310 --out_nodes="labels;dets"
if [ -f "fcos_bs1.om" ] && [ -f "fcos_bs16.om" ]; then
    echo "success"
else
    echo "fail!"
fi