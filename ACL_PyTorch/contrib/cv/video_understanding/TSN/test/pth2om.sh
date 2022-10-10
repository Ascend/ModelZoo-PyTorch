#!/bin/bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python3.7 pytorch2onnx.py ./mmaction2/configs/recognition/tsn/tsn_r50_1x1x3_75e_ucf101_rgb.py ./tsn_r50_1x1x3_75e_ucf101_rgb_20201023-d85ab600.pth --verify

mkdir -p om
atc --framework=5 --model=tsn.onnx --output=tsn_1 --input_format=NCDHW --input_shape="image:1,75,3,256,256" --log=debug --soc_version=Ascend310 --auto_tune_mode "RL,GA"
atc --framework=5 --model=tsn.onnx --output=tsn_4 --input_format=NCDHW --input_shape="image:4,75,3,256,256" --log=debug --soc_version=Ascend310 --auto_tune_mode "RL,GA"
atc --framework=5 --model=tsn.onnx --output=tsn_8 --input_format=NCDHW --input_shape="image:8,75,3,256,256" --log=debug --soc_version=Ascend310 --auto_tune_mode "RL,GA"
if [ -f "om/tsm_bs1.om" ] && [ -f "om/tsm_bs16.om" ]; then
    echo "success"
else
    echo "fail!"
fi