#!/bin/bash

rm -rf AdvancedEAST_dybs.onnx
python3.7 AdvancedEAST_pth2onnx.py 3T736_best_mF1_score.pth AdvancedEAST_dybs.onnx

source env.sh
rm -rf AdvancedEAST_bs1.om AdvancedEAST_bs16.om
atc --framework=5 --model=AdvancedEAST_dybs.onnx --output=AdvancedEAST_bs1 --input_format=NCHW --input_shape='input_1:1,3,736,736' --log=debug --soc_version=Ascend310 --auto_tune_mode='RL,GA'
atc --framework=5 --model=AdvancedEAST_dybs.onnx --output=AdvancedEAST_bs16 --input_format=NCHW --input_shape='input_1:16,3,736,736' --log=debug --soc_version=Ascend310 --auto_tune_mode='RL,GA'
if [ -f "AdvancedEAST_bs1.om" ] && [ -f "AdvancedEAST_bs16.om" ]; then
    echo "success"
else
    echo "fail!"
fi
