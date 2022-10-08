#!/usr/bin/env bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python3.7.5 -u pth2onnx.py

# onnx2om
#  --input_format=NCHW
atc --framework=5 --model=gaitset_submit.onnx --output=gaitset_submit --input_shape="image_seq:1,100,64,44" --log=debug --soc_version=Ascend310
