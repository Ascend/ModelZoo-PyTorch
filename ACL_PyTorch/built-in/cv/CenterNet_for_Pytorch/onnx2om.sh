#!/bin/bash
clear

source /usr/local/Ascend/ascend-toolkit/set_env.sh

atc --framework=5 --model=../models/ctdet_coco_dla_2x_modify.onnx --output=../models/ctdet_coco_dla_2x \
--input_format=NCHW --input_shape=image:1,3,512,512 --log=error --soc_version=Ascend310

rm -rf fusion_result.json kernel_meta