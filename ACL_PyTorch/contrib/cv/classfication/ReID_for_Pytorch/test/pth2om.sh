#!/bin/bash

rm -rf ReID.onnx
python3.7 ReID_pth2onnx.py --config_file='reid-strong-baseline/configs/softmax_triplet_with_center.yml' MODEL.PRETRAIN_CHOICE "('self')" TEST.WEIGHT "('market_resnet50_model_120_rank1_945.pth')" TEST.NECK_FEAT "('before')" TEST.FEAT_NORM "('no')"
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

rm -rf ReID_bs1.om ReID_bs16.om
source env.sh
atc --framework=5 --model=ReID.onnx --output=ReID_bs1 --input_format=NCHW --input_shape="image:1,3,256,128" --log=debug --soc_version=Ascend310
atc --framework=5 --model=ReID.onnx --output=ReID_bs16 --input_format=NCHW --input_shape="image:16,3,256,128" --log=debug --soc_version=Ascend310 --auto_tune_mode="RL,GA"

if [ -f "ReID_bs1.om" ] && [ -f "ReID_bs16.om" ]; then
    echo "success"
else
    echo "fail!"
fi