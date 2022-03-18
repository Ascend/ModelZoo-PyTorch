#!/bin/bash

rm -rf pnasnet5large.onnx
rm -rf pnasnet5large_sim_bs1.onnx
rm -rf pnasnet5large_sim_bs16.onnx
python3.7 pnasnet5large_onnx.py pnasnet5large.onnx
python3.7 -m onnxsim  --input-shape="1,3,331,331" pnasnet5large.onnx pnasnet5large_sim_bs1.onnx
python3.7 -m onnxsim  --input-shape="16,3,331,331" pnasnet5large.onnx pnasnet5large_sim_bs16.onnx
source env.sh
rm -rf pnasnet5large_bs1.om pnasnet5large_bs16.om
atc --framework=5 --model=./pnasnet5large_sim_bs1.onnx --input_format=NCHW --input_shape="image:1,3,331,331" --output=pnasnet5large_bs1 --log=debug --soc_version=Ascend310
atc --framework=5 --model=./pnasnet5large_sim_bs16.onnx --input_format=NCHW --input_shape="image:16,3,331,331" --output=pnasnet5large_bs16 --log=debug --soc_version=Ascend310
if [ -f "pnasnet5large_bs1.om" ] && [ -f "pnasnet5large_bs16.om" ]; then
    echo "success"
else
    echo "fail!"
fi