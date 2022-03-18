#!/bin/bash

rm -rf setr_naive_768x768_bs1.onnx

python3.7  SETR/tools/pytorch2onnx.py SETR/configs/SETR/SETR_Naive_768x768_40k_cityscapes_bs_8.py \
        --checkpoint SETR/author_pth/SETR_Naive_cityscapes_b8_40k.pth \
        --shape 768 768 \
        --output-file setr_naive_768x768_bs1.onnx
rm -rf setr_naive_768x768_sim.onnx
python3.7 -m onnxsim setr_naive_768x768_bs1.onnx setr_naive_768x768_sim.onnx \
        --input-shape=1,3,768,768 
source env.sh
rm -rf setr_naive_768x768_bs1.om
atc --framework=5 --model=setr_naive_768x768_sim.onnx --output=setr_naive_768x768_bs1 \
    --input_format=NCHW --input_shape="img:1,3,768,768" --log=debug --soc_version=Ascend310
