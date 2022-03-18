#!/bin/bash
clear

export install_path=/home/zdy/Ascend_0706/ascend-toolkit/latest
export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export PYTHONPATH=${install_path}/atc/python/site-packages:${install_path}/pyACL/python/site-packages/acl:$PYTHONPATH
export LD_LIBRARY_PATH=${install_path}/atc/lib64:${install_path}/acllib/lib64:$LD_LIBRARY_PATH
export ASCEND_OPP_PATH=${install_path}/opp

atc --framework=5 --model=../models/ctdet_coco_dla_2x_modify.onnx --output=../models/ctdet_coco_dla_2x \
--input_format=NCHW --input_shape=image:1,3,512,512 --log=error --soc_version=Ascend310

rm -rf fusion_result.json kernel_meta