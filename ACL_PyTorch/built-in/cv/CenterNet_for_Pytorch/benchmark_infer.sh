#!/bin/bash
clear

source /usr/local/Ascend/ascend-toolkit/set_env.sh

./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=1 -om_path=../models/ctdet_coco_dla_2x.om \
-input_text_path=./pre_bin/bin_file.info -input_width=512 -input_height=512 -output_binary=true -useDvpp=false
