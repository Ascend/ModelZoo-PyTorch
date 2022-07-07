#!/bin/bash

rm PAMTRI_bs1.onnx
python3.7 PAMTRI_pth2onnx.py -d veri -a densenet121 --root /opt/npu --load-weights models/densenet121-xent-htri-veri-multitask/model_best.pth.tar --output_path ./PAMTRI_bs1.onnx --multitask --batch_size 1
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

rm PAMTRI_bs16.onnx
python3.7 PAMTRI_pth2onnx.py -d veri -a densenet121 --root /opt/npu --load-weights models/densenet121-xent-htri-veri-multitask/model_best.pth.tar --output_path ./PAMTRI_bs16.onnx --multitask --batch_size 16
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

rm PAMTRI_bs1.om PAMTRI_bs16.om
source /usr/local/Ascend/ascend-toolkit/set_env.sh
atc --framework=5 --model=PAMTRI_bs1.onnx --output=PAMTRI_bs1 --input_format=NCHW --input_shape="input.1:1,3,256,256" --log=debug --soc_version=Ascend710
atc --framework=5 --model=PAMTRI_bs16.onnx --output=PAMTRI_bs16 --input_format=NCHW --input_shape="input.1:16,3,256,256" --log=debug --soc_version=Ascend710

if [ -f "PAMTRI_bs1.om" ] && [ -f "PAMTRI_bs16.om" ]; then
    echo "success"
else
    echo "fail!"
fi