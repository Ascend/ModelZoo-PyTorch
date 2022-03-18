#!/bin/bash

rm -rf fcn_r50-d8_512x512_20k_voc12aug.onnx
python3.7 mmsegmentation/tools/pytorch2onnx.py mmsegmentation/configs/fcn/fcn_r50-d8_512x512_20k_voc12aug.py --checkpoint fcn_r50-d8_512x512_20k_voc12aug_20200617_010715-52dc5306.pth --output-file fcn_r50-d8_512x512_20k_voc12aug.onnx --shape 500 500
source env.sh
rm -rf fcn_r50-d8_512x512_20k_voc12aug_bs1.om
atc --framework=5 --model=fcn_r50-d8_512x512_20k_voc12aug.onnx  --output=fcn_r50-d8_512x512_20k_voc12aug_bs1 --input_format=NCHW --input_shape=" input:1,3,500,500" --log=debug --soc_version=Ascend310
if [ -f "fcn_r50-d8_512x512_20k_voc12aug_bs1.om" ] ; then
    echo "success"
else
    echo "fail!"
fi