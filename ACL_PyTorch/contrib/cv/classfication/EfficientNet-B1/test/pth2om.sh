#!/bin/bash

rm -rf efficientnetB1.onnx 
python3.7 Efficient-B1_pth2onnx.py EN-B1_dds_8gpu.pyth ./pycls/configs/dds_baselines/effnet/EN-B1_dds_8gpu.yaml efficientnetB1.onnx 
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

rm -rf Efficient-b1_bs1.om Efficient-b1_bs16.om
source /usr/local/Ascend/ascend-toolkit/set_env.sh
atc --framework=5 --model=efficientnetB1.onnx  --output=Efficient-b1_bs1 --input_format=NCHW --input_shape="image:1,3,240,240" --log=debug --soc_version=Ascend310
atc --framework=5 --model=efficientnetB1.onnx  --output=Efficient-b1_bs16 --input_format=NCHW --input_shape="image:16,3,240,240" --log=debug --soc_version=Ascend310 --auto_tune_mode="RL,GA"

if [ -f "Efficient-b1_bs1.om" ] && [ -f "Efficient-b1_bs1.om" ]; then
    echo "success"
else
    echo "fail!"
fi
