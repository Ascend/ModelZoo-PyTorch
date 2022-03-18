#!/bin/bash

source scripts/set_npu_env.sh

#python pro_train.py 
python pth2onnx.py ./net.pth DnCNN-S-15.onnx 
