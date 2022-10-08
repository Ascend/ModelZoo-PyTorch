#!/bin/bash

datasets_path="mmaction2/data/kinetics400"

for para in $*
do
    if [[ $para == --datasets_path* ]]; then
        datasets_path=`echo ${para#*=}`
    fi
done

python3.7 slowfast_k400_preprocess.py --config mmaction2/configs/recognition/slowfast/slowfast_r50_8x8x1_256e_kinetics400_rgb.py --batch_size 1 --data_root ${datasets_path}/videos_val/ --ann_file ${datasets_path}/kinetics400_val_list_videos.txt --name out_bin_1
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
python3.7 slowfast_k400_preprocess.py --config mmaction2/configs/recognition/slowfast/slowfast_r50_8x8x1_256e_kinetics400_rgb.py --batch_size 16 --data_root ${datasets_path}/videos_val/ --ann_file ${datasets_path}/kinetics400_val_list_videos.txt --name out_bin_16
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

source /usr/local/Ascend/ascend-toolkit/set_env.sh

mkdir -p output/out_bs1
rm -rf output/out_bs1/*
./tools/msame/out/msame --model ./om/slowfast_bs1.om --input ${datasets_path}/out_bin_1 --output ./output/out_bs1/ --outfmt TXT
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

mkdir -p output/out_bs16
rm -rf output/out_bs16/*
./tools/msame/out/msame --model ./om/slowfast_bs16.om --input ${datasets_path}/out_bin_16 --output ./output/out_bs16/ --outfmt TXT
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

mkdir -p result
./benchmark.x86_64 -round=20 -om_path=./om/slowfast_bs1.om -device_id=0 -batch_size=1
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
./benchmark.x86_64 -round=20 -om_path=./om/slowfast_bs16.om -device_id=0 -batch_size=16
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

echo "====accuracy data===="
python3.7 slowfast_k400_postprocess.py --result_path ./output/out_bs1 --info_path ${datasets_path}/k400.info
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
python3.7 slowfast_k400_postprocess.py --result_path ./output/out_bs16  --info_path ${datasets_path}/k400.info
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

echo "====performance data===="
python3.7 test/parse.py result/PureInfer_perf_of_slowfast_bs1_in_device_0.txt
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
python3.7 test/parse.py result/PureInfer_perf_of_slowfast_bs16_in_device_0.txt
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "success"
