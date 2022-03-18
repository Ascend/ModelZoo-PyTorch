#!/bin/bash

datasets_path="/opt/npu"

for para in $*
do
    if [[ $para == --datasets_path* ]]; then
        datasets_path=`echo ${para#*=}`
    fi
done

arch=`uname -m`
source env.sh
rm -rf ./result/*

echo "====conducting preprocess, inferrence and postprocess===="
python3.7 wholeprocess.py ${datasets_path}/OTB/ ./pre_dataset ./dataset_info $(arch) 0
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

echo "====evaluating performance===="
rm -rf ./pre_dataset1
rm -rf ./pre_dataset2
rm -rf ./dataset1.info
rm -rf ./dataset2.info

python3.7 get_perf_data.py ./pre_dataset1 ./pre_dataset2 ./dataset1.info ./dataset2.info
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo '==> 1. creating data for performance evaluation successfully.'

./benchmark.${arch} -model_type=vision -device_id=1 -batch_size=1 -om_path=./om/exemplar_bs1.om -input_text_path=./dataset1.info -input_width=127 -input_height=127 -output_binary=True -useDvpp=False >/dev/null 2>&1
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo '==> 2. conducting exemplar_bs1.om on device 1 successfully.'

./benchmark.${arch} -model_type=vision -device_id=2 -batch_size=1 -om_path=./om/search_bs1.om -input_text_path=./dataset2.info -input_width=255 -input_height=255 -output_binary=True -useDvpp=False >/dev/null 2>&1
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo '==> 3. conducting search_bs1.om on device 2 successfully.'

echo "====performance data of exemplar branch===="
python3.7 test/parse.py ./result/perf_vision_batchsize_1_device_1.txt
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

echo "====performance data of search branch===="
python3.7 test/parse.py ./result/perf_vision_batchsize_1_device_2.txt
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

echo "success"