#!/bin/bash

datasets_path="/root/datasets/"

for para in $*
do
    if [[ $para == --datasets_path* ]]; then
        datasets_path=`echo ${para#*=}`
    fi
done

arch=`uname -m`
rm -rf ./prep_dataset_query
rm -rf ./prep_dataset_gallery
python3.7 ReID_preprocess.py ${datasets_path}/market1501/query ./prep_dataset_query
python3.7 ReID_preprocess.py ${datasets_path}/market1501/bounding_box_test ./prep_dataset_gallery
mv prep_dataset_gallery/* prep_dataset_query/
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
python3.7 gen_dataset_info.py bin ./prep_dataset_query ./prep_bin.info 128 256
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
source env.sh
rm -rf result/dumpOutput_device0
./benchmark.${arch} -model_type=vision -device_id=0 -batch_size=1 -om_path=./ReID_bs1.om -input_text_path=./prep_bin.info -input_width=128 -input_height=256 -output_binary=True -useDvpp=False
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
rm -rf result/dumpOutput_device1
./benchmark.${arch} -model_type=vision -device_id=1 -batch_size=16 -om_path=./ReID_bs16.om -input_text_path=./prep_bin.info -input_width=128 -input_height=256 -output_binary=True -useDvpp=False
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
python3.7 ReID_postprocess.py --query_dir=${datasets_path}/market1501/query --gallery_dir=${datasets_path}/market1501/bounding_box_test --pred_dir=./result/dumpOutput_device0 > result_bs1.json
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
python3.7 ReID_postprocess.py --query_dir=${datasets_path}/market1501/query --gallery_dir=${datasets_path}/market1501/bounding_box_test --pred_dir=./result/dumpOutput_device1 > result_bs16.json
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "====accuracy data===="
python3.7 test/parse.py result_bs1.json
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
python3.7 test/parse.py result_bs16.json
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
python3.7 test/parse.py result/perf_vision_batchsize_16_device_1.txt
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "success"