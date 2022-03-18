#!/bin/bash

rm -rf rcf.onnx
python3.7 rcf_pth2onnx.py --pth_path RCF-pytorch/RCFcheckpoint_epoch12.pth --onnx_name rcf_bs1 \
--batch_size 1 1 --height 321 481 --width 481 321
if [ $? != 0 ]; then
    echo "pth to onnx fail!"
    exit -1
else
    echo "pth to onnx success!"
fi

python3.7 change_model.py --input_name rcf_bs1 --output_name rcf_bs1_change \
--simplified_name rcf_bs1_change_sim --batch_size 1 1 --height 321 481 --width 481 321
if [ $? != 0 ]; then
    echo "change and simplify onnx model fail!"
    exit -1
else
    echo "change and simplify onnx model success!"
fi
