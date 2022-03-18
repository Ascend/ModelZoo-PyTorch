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
python3.7 srcnn_preprocess.py -s ${datasets_path}/set5 -d ./prep_data
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
python3.7 get_info.py bin ./prep_data/bin ./srcnn_prep_bin.info 256 256
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
source env.sh
rm -rf result/dumpOutput_device0
./benchmark -model_type=vision -device_id=0 -batch_size=1 -om_path=./srcnn_x2.om -input_text_path=./srcnn_prep_bin.info -input_width=256 -input_height=256 -output_binary=False -useDvpp=False
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "====accuracy data===="
python3.7 srcnn_postprocess.py --res ./result/dumpOutput_device0/ --png_src ./prep_data/png --bin_src ./prep_data/bin --save ./result/save
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "====performance data===="
./benchmark -round=20 -om_path=./srcnn_x2.om -device_id=0 -batch_size=1
python3.7 test/parse.py result/perf_vision_batchsize_1_device_0.txt
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "success" 
