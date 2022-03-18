#!/bin/bash

datasets_path="./SRFlow/datasets/"

for para in $*
do
    if [[ $para == --datasets_path* ]]; then
        datasets_path=`echo ${para#*=}`
    fi
done

arch=`uname -m`
rm -rf ./prep_data
python3.7 srflow_preprocess.py -s $datasets_path/div2k-validation-modcrop8-x8 -d ./prep_data
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
python3.7 get_info.py bin ./prep_data/bin ./srflow_prep_bin.info 256 256
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
source env.sh
rm -rf result/dumpOutput_device0
./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=1 -om_path=./srflow_df2k_x8_bs1.om -input_text_path=./srflow_prep_bin.info -input_width=256 -input_height=256 -output_binary=True -useDvpp=False
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "====accuracy data===="
python3.7 srflow_postprocess.py --hr $datasets_path/div2k-validation-modcrop8-gt/ --binres ./result/dumpOutput_device0/  --save ./result/save/
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "====performance data===="
python3.7 test/parse.py result/perf_vision_batchsize_1_device_0.txt
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "success" 
