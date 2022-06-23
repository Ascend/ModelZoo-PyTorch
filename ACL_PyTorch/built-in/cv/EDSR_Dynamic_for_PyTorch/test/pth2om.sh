#!/bin/bash


# pth to onnx
python3 pth2onnx.py --scale 2 --n_resblock 32 --n_feats 256 --res_scale 0.1 --input_path $1

source /usr/local/Ascend/ascend-toolkit/set_env.sh
# onnx to om
mkdir -p models/om
atc --model=models/onnx/EDSR_x2.onnx --framework=5 --output=models/om/EDSR_x2 --input_format=ND --log=debug --soc_version=$2 --input_fp16_nodes="image" --output_type=FP16 --input_shape_range="image:[1,3,100~1080,100~1920]"

