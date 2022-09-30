#!/bin/bash

python3.7 ./pth2onnx.py ./model/model.pt ./model/model_mkt1501_bs1.onnx 1
python3.7 ./pth2onnx.py ./model/model.pt ./model/model_mkt1501_bs16.onnx 16
#env.sh
source /usr/local/Ascend/ascend-toolkit/set_env.sh

rm -rf ./mgn_mkt1501_bs1.om ./mgn_mkt1501_bs16.om

atc --framework=5 --model=./model/model_mkt1501_bs1.onnx --input_format=NCHW --input_shape="image:1,3,384,128" --output=mgn_mkt1501_bs1 --log=debug --soc_version=Ascend310
atc --framework=5 --model=./model/model_mkt1501_bs16.onnx --input_format=NCHW --input_shape="image:16,3,384,128" --output=mgn_mkt1501_bs16 --log=debug --soc_version=Ascend310
if [ -f "./mgn_mkt1501_bs1.om" ] && [ -f "./mgn_mkt1501_bs16.om" ];then
    echo "success"
else
    echo "fail!"
fi