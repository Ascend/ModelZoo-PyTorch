#!/bin/bash

source test/env_npu.sh

#python pro_train.py 
python pth2onnx.py ./net.pth DnCNN-S-15.onnx 
