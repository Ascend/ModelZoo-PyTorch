#!/bin/bash

python3.7 faceboxes_pth2onnx.py  --trained_model weights/FaceBoxesProd.pth --save_folder faceboxes-b0.onnx
pip3.7 install onnx-simplifier
python3.7 -m onnxsim --input-shape="1,3,1024,1024" --dynamic-input-shape faceboxes-b0.onnx faceboxes-b0_sim.onnx


rm -rf ./faceboxes-b0_bs1.om ./faceboxes-b0_bs16.om
export REPEAT_TUNE=True
atc --framework=5 --model=faceboxes-b0_sim.onnx --output=faceboxes-b0_bs1 --input_format=NCHW --input_shape="image:1,3,1024,1024" --log=debug --soc_version=Ascend310 --out_nodes="Reshape_127:0;Softmax_134:0" --auto_tune_mode="RL,GA" 
atc --framework=5 --model=faceboxes-b0_sim.onnx --output=faceboxes-b0_bs16 --input_format=NCHW --input_shape="image:16,3,1024,1024" --log=debug --soc_version=Ascend310 --out_nodes="Softmax_134:0;Reshape_127:0" --auto_tune_mode="RL,GA"

if [ -f "./faceboxes-b0_bs1.om" ] && [ -f "./faceboxes-b0_bs16.om" ];then
    echo "success"
else
    echo "fail!"
fi