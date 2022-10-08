#!/bin/bash

datasets_path="/opt/npu/"

for para in $*
do
    if [[ $para == --datasets_path* ]]; then
        datasets_path=`echo ${para#*=}`
    fi
done

python3.7 tsn_ucf101_preprocess.py --batch_size 1 --data_root ${datasets_path}/ucf101 --name out_bin_1
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
python3.7 tsn_ucf101_preprocess.py --batch_size 8 --data_root ${datasets_path}/ucf101 --name out_bin_8
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

source /usr/local/Ascend/ascend-toolkit/set_env.sh
mkdir -p output/out_bs1
rm -rf output/out_bs1/*
./tools/msame/out/msame --model ./om/tsn_1.om --input /opt/npu/ucf101/out_bin_1 --output ./output/out_bs1/ --outfmt TXT
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
mkdir -p output/out_bs8
rm -rf output/out_bs8/*
./tools/msame/out/msame --model ./om/tsn_8.om --input /opt/npu/ucf101/out_bin_8 --output ./output/out_bs8/ --outfmt TXT
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

mkdir -p result
./benchmark.x86_64 -round=20 -om_path=./om/tsn_1.om -device_id=0 -batch_size=1
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
./benchmark.x86_64 -round=20 -om_path=./om/tsn_8.om -device_id=0 -batch_size=8
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

echo "====accuracy data===="
python3.7 tsn_ucf101_postprocess.py --result_path ./output/out_bs1 --info_path ${datasets_path}/ucf101/ucf101_1.info --batch_size 1
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
python3.7 tsn_ucf101_postprocess.py --result_path ./output/out_bs8 --info_path ${datasets_path}/ucf101/ucf101_8.info --batch_size 8
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

echo "====performance data===="
python3.7 test/parse.py result/PureInfer_perf_of_tsn_1_in_device_0.txt
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
python3.7 test/parse.py result/PureInfer_perf_of_tsn_8_in_device_0.txt
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "success"