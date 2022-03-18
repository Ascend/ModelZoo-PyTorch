#!/bin/bash
set -eu


# set env
source env.sh
# convert onnx
python3.7 pth2onnx.py


device_id=0
cur_dir=`pwd`


for bs in 1 4 8 16 32 64;
do
    python3.7 -m onnxsim models/u2net.onnx models/u2net_sim_bs${bs}.onnx --input-shape "image:${bs},3,320,320" &> log
    python3.7 fix_onnx.py models/u2net_sim_bs${bs}.onnx models/u2net_sim_bs${bs}_fixv2.onnx &> log
    atc --framework=5 --model=models/u2net_sim_bs${bs}_fixv2.onnx --output=models/u2net_sim_bs${bs}_fixv2 --input_format=NCHW --input_shape="image:${bs},3,320,320" --out_nodes='Sigmoid_1048:0' --log=error --soc_version=Ascend710
done
