#!/bin/bash

datasets_path="/root/datasets/"

for para in $*
do
    if [[ $para == --datasets_path* ]]; then
        datasets_path=`echo ${para#*=}`
    fi
done

arch=`uname -m`
rm -rf ./prep_dataset
python3.7 RDN_preprocess.py --src-path=${datasets_path}/set5 --save-path=./prep_dataset --scale=2

if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
python3.7 gen_dataset_info.py bin ./prep_dataset/data ./RDN_prep_bin.info 114 114
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
source /usr/local/Ascend/ascend-toolkit/set_env.sh
rm -rf result/dumpOutput_device0
./benchmark.${arch} -model_type=vision -device_id=0 -batch_size=1 -om_path=./rdn_x2_bs1.om -input_text_path=./RDN_prep_bin.info -input_width=114 -input_height=114 -output_binary=False -useDvpp=False
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
python3.7 RDN_postprocess.py --pred-path=./result/dumpOutput_device0 --label-path=./prep_dataset/label --result-path=./result.json --width=114 --height=114 --scale=2
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo -e "\n====accuracy data===="
python3.7 test/parse.py result.json
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo -e "====performance data===="
python3.7 test/parse.py ./result/perf_vision_batchsize_1_device_0.txt
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "success"
