#!/bin/bash
clear

export install_path=/home/zdy/Ascend_0706/ascend-toolkit/latest
export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export PYTHONPATH=${install_path}/atc/python/site-packages:${install_path}/pyACL/python/site-packages/acl:$PYTHONPATH
export LD_LIBRARY_PATH=${install_path}/atc/lib64:${install_path}/acllib/lib64:$LD_LIBRARY_PATH
export ASCEND_OPP_PATH=${install_path}/opp

./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=1 -om_path=../models/ctdet_coco_dla_2x.om \
-input_text_path=./pre_bin/bin_file.info -input_width=512 -input_height=512 -output_binary=true -useDvpp=false
