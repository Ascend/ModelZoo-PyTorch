#!/bin/bash
export MSPN_HOME=$(pwd)
export PYTHONPATH=$PYTHONPATH:$MSPN_HOME
rm -rf MSPN.onnx
python3.7 MSPN_pth2onnx.py 
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

rm -rf MSPN_bs1.om MSPN_bs16.om
source env.sh
atc --framework=5 --model=MSPN.onnx --output=MSPN_bs1 --input_format=NCHW --input_shape="input:1,3,256,192" --log=debug --soc_version=Ascend310
atc --framework=5 --model=MSPN.onnx --output=MSPN_bs16 --input_format=NCHW --input_shape="input:16,3,256,192" --log=debug --soc_version=Ascend310 

if [ -f "MSPN_bs1.om" ] && [ -f "MSPN_bs16.om" ]; then
    echo "success"
else
    echo "fail!"
fi