#!/bin/bash

datasets_path="/root/datasets/"

for para in $*
do
    if [[ $para == --datasets_path* ]]; then
        datasets_path=`echo ${para#*=}`
    fi
done
arch=`uname -m`
rm -rf ./pre_dataset
echo "preprocess"
python3.7 SSD_MobileNet_preprocess.py ${datasets_path}/VOC2007/JPEGImages ./pre_dataset
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

echo "get_info"
python3.7 get_info.py bin pre_dataset mb-ssd_prep_bin.info 300 300
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

source env.sh
rm -rf result/dumpOutput_device0
echo "benchmark.x86_64"
./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=1 -om_path=mb1-ssd_bs1.om -input_text_path=mb-ssd_prep_bin.info -input_width=300 -input_height=300 -output_binary=true -useDvpp=False
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi


rm -rf result/dumpOutput_device1
echo "benchmark.x86_64"
./benchmark.x86_64 -model_type=vision -device_id=1 -batch_size=16 -om_path=mb1-ssd_bs16.om -input_text_path=mb-ssd_prep_bin.info -input_width=300 -input_height=300 -output_binary=true -useDvpp=False
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

echo "eval_acc_bs1"
python3.7 SSD_MobileNet_postprocess.py ${datasets_path}/VOC2007/ voc-model-labels.txt ./result/dumpOutput_device0/ ./eval_results0/

echo "eval_acc_bs16"
python3.7 SSD_MobileNet_postprocess.py ${datasets_path}/VOC2007/ voc-model-labels.txt ./result/dumpOutput_device1/ ./eval_results1/
