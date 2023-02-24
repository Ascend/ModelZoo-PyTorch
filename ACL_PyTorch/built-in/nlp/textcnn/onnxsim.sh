#!/bin/bash

if [ ! -d "./onnx_sim_dir" ]
then
  mkdir ./onnx_sim_dir
fi

for i in 1 4 8 16 32 64
do
  python3 -m onnxsim --input-shape="sentence:${i},32" ./ascend_textcnn/dy_textcnn.onnx ./onnx_sim_dir/textcnn_${i}bs_sim.onnx
done
