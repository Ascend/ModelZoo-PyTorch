#!/bin/bash

datasets_path="./data/DIV2K"

for para in $*
do
    if [[ $para == --datasets_path* ]]; then
        datasets_path=`echo ${para#*=}`
    fi
done
arch=`uname -m`
rm -rf ./DIV2K_valid_LR_bicubic_bin/X2/
python3.7 Wdsr_prePorcess.py --lr_path ${datasets_path}/DIV2K_valid_LR_bicubic/X2/ --hr_path ${datasets_path}/DIV2K_valid_HR/ --save_lr_path ./DIV2K_valid_LR_bicubic_bin/X2/  --width 1020 --height 1020 --scale 2
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
python3.7 get_info.py bin ./DIV2K_valid_LR_bicubic_bin/X2/ wdsr_bin.info 1020 1020
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
source env.sh
rm -rf result/dumpOutput_device3
./benchmark.x86_64 -model_type=vision -device_id=3 -batch_size=1 -om_path=./wdsr_bs1.om -input_text_path=./wdsr_bin.info -input_width=1020 -input_height=1020 -output_binary=True -useDvpp=False
rm -fr result/dumpOutput_device2
./benchmark.x86_64 -model_type=vision -device_id=2 -batch_size=8 -om_path=./wdsr_bs8.om -input_text_path=./wdsr_bin.info -input_width=1020 -input_height=1020 -output_binary=True -useDvpp=False
echo "====accuracy data===="
python3.7 Wdsr_postProcess.py --bin_data_path ./result/dumpOutput_device3/ --dataset_path ${datasets_path}/DIV2K_valid_HR/ --result result_bs1.txt --scale 2
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
python3.7 Wdsr_postProcess.py --bin_data_path ./result/dumpOutput_device2/ --dataset_path ${datasets_path}/DIV2K_valid_HR/ --result result_bs8.txt --scale 2
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "====performance data===="
python3.7 test/parse.py result/perf_vision_batchsize_1_device_3.txt
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
python3.7 test/parse.py result/perf_vision_batchsize_8_device_2.txt
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "success"