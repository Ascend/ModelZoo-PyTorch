#!/bin/bash

# Copyright 2019 Mobvoi Inc. All Rights Reserved.
. ./path.sh || exit 1;

yaml_path=$1
decode_checkpoint=$2

mkdir onnx
python3 wenet/bin/export_onnx.py \
    --config $yaml_path \
    --checkpoint $decode_checkpoint \
    --output_onnx_file onnx
