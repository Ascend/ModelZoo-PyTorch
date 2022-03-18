#!/bin/bash

datasets_path="./dataset/"
for para in $*
do
    if [[ $para == --datasets_path* ]]; then
        datasets_path=`echo ${para#*=}`
    fi
done

arch=`uname -m`

rm -rf ./pre_dataset
python3.7 ./fairmot_preprocess.py --data_root=${datasets_path} --output_dir=./pre_dataset
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
rm -rf seq.info
python3.7 ./get_dataset_info.py --file_path=./pre_dataset --file_name=./seq.info --width=1088 --height=608

if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

# source ../env.sh
rm -rf result
./benchmark.${arch} -model_type=vision -device_id=0 -batch_size=1 -om_path=fairmot_bs1.om -input_text_path=seq.info -input_width=1088 -input_height=608 -output_binary=True -useDvpp=False
mv result/dumpOutput_device0 result/dumpOutput_device0_bs1
./benchmark.${arch} -model_type=vision -device_id=0 -batch_size=8 -om_path=fairmot_bs8.om -input_text_path=seq.info -input_width=1088 -input_height=608 -output_binary=True -useDvpp=False
mv result/dumpOutput_device0 result/dumpOutput_device0_bs8
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

python3.7 ./fairmot_postprocess.py --data_dir=${datasets_path}  --input_root=./result/dumpOutput_device0_bs1 > bs_1_result.log
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
python3.7 ./fairmot_postprocess.py --data_dir=${datasets_path}  --input_root=./result/dumpOutput_device0_bs8 > bs_8_result.log
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "====accuracy data===="
python3.7 test/parse.py bs_1_result.log
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
python3.7 test/parse.py bs_8_result.log
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
python3.7 test/parse.py result/perf_vision_batchsize_8_device_0.txt
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

echo "success"
