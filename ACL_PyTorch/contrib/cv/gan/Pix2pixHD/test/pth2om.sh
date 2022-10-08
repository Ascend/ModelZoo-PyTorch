#!/bin/bash

rm -rf pix2pixhd.onnx
python pix2pixhd_pth2onnx.py --load_pretrain ./pix2pixHD/checkpoints/label2city_1024p --output_file pix2pixhd.onnx
source /usr/local/Ascend/ascend-toolkit/set_env.sh
rm -rf pix2pixhd_bs1.om
atc --framework=5 --model=./pix2pixhd.onnx --input_format=NCHW --input_shape="input_concat:1,36,1024,2048" --output=pix2pixhd_bs1 --log=debug --soc_version=Ascend310
if [ -f "pix2pixhd_bs1.om" ] ; then
    echo "success"
else
    echo "fail!"
fi