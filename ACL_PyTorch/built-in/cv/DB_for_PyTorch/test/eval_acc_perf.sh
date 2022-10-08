#!/bin/bash

datasets_path="./datasets/"

for para in $*
do
    if [[ $para == --datasets_path* ]]; then
        datasets_path=`echo ${para#*=}`
    fi
done

arch=`uname -m`
cd DB/
rm -rf ./prep_dataset
python3.7 ../db_preprocess.py --image_src_path=${datasets_path}/icdar2015/test_images --bin_file_path=./prep_dataset
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
python3.7 ../gen_dataset_info.py bin ./prep_dataset ./db_dataset.info 1280 736
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
source /usr/local/Ascend/ascend-toolkit/set_env.sh
rm -rf result/dumpOutput_device0
./benchmark.${arch} -model_type=vision -device_id=0 -batch_size=1 -om_path=db_bs1.om -input_text_path=./db_dataset.info -input_width=1280 -input_height=736 -useDvpp=False -output_binary=True
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
rm -rf result/dumpOutput_device1
./benchmark.${arch} -model_type=vision -device_id=1 -batch_size=16 -om_path=db_bs16.om -input_text_path=./db_dataset.info -input_width=1280 -input_height=736 -useDvpp=False -output_binary=True
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
python3.7 ../db_postprocess.py experiments/seg_detector/ic15_resnet50_deform_thre.yaml --bin_data_path ./result/dumpOutput_device0/ --box_thresh 0.6 > result_bs1.json
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
python3.7 ../db_postprocess.py experiments/seg_detector/ic15_resnet50_deform_thre.yaml --bin_data_path ./result/dumpOutput_device1/ --box_thresh 0.6 > result_bs16.json
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "====accuracy data===="
python3.7 ../test/parse.py result_bs1.json
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
python3.7 ../test/parse.py result_bs16.json
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "====performance data===="
python3.7 ../test/parse.py result/perf_vision_batchsize_1_device_0.txt
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
python3.7 ../test/parse.py result/perf_vision_batchsize_16_device_1.txt
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "success"
