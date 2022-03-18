#!/bin/bash

datasets_path="/root/datasets"

for para in $*
do
    if [[ $para == --datasets_path* ]]; then
        datasets_path=`echo ${para#*=}`
    fi
done

arch=`uname -m`
rm -rf ./prep_dataset
python3.7 edsr_preprocess.py -s ${datasets_path}/div2k/LR -d ./prep_data
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
python3.7 get_info.py bin ./prep_data/bin ./edsr_prep_bin.info 1020 1020
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
source env.sh
rm -rf result/dumpOutput_device0
./benchmark -model_type=vision -device_id=0 -batch_size=1 -om_path=./edsr_x2.om -input_text_path=./edsr_prep_bin.info -input_width=1020 -input_height=1020 -output_binary=True -useDvpp=False
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "====accuracy data===="
python3.7 edsr_postprocess.py --res ./result/dumpOutput_device0/ --HR ${datasets_path}/div2k/HR
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "====performance data===="
python3.7 edsr_pth2onnx.py --pth edsr_x2.pt --onnx edsr_x2_256.onnx --size 256
atc --framework=5 --model=edsr_x2_256.onnx --output=edsr_x2_256 --input_format=NCHW --input_shape="input.1:1,3,256,256" --log=debug --soc_version=Ascend310 --fusion_switch_file=switch.cfg
./benchmark -round=20 -om_path=./edsr_x2_256.om -device_id=0 -batch_size=1

if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "success" 
