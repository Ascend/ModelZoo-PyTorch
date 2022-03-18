#!/bin/bash

datasets_path=""
arch=`uname -m`

for para in $*
do
    if [[ $para == --datasets_path* ]]; then
        datasets_path=`echo ${para#*=}`
    fi
done

python LV_Vit_preprocess.py --src_path ${datasets_path} --save_path ./data/prep_dataset;
python gen_dataset_info.py ./data/prep_dataset/ ./lvvit_prep_bin.info;

source ./env.sh;

rm -rf result/dumpOutput_device0;
rm -rf result/dumpOutput_device0_bs1;
rm -rf result/dumpOutput_device0_bs16;

./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=1 -om_path=./model/model_best_bs1_sim.om -input_text_path=lvvit_prep_bin.info -input_width=224 -input_height=224 -output_binary=False -useDvpp=False
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
mv result/dumpOutput_device0 result/dumpOutput_device0_bs1

./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=16 -om_path=./model/model_best_bs16_sim.om -input_text_path=lvvit_prep_bin.info -input_width=224 -input_height=224 -output_binary=False -useDvpp=False
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
mv result/dumpOutput_device0 result/dumpOutput_device0_bs16

echo "====accuracy data===="

python LV_Vit_postprocess.py result/dumpOutput_device0_bs1/ ./data/val.txt;
python LV_Vit_postprocess.py result/dumpOutput_device0_bs16/ ./data/val.txt;

echo "====performance data===="

bs1_one=`cat result/perf_vision_batchsize_1_device_0.txt | grep 'Interface throughputRate' | awk '{print $6}'`
bs1_fps=`awk 'BEGIN{printf "%.2f\n", 4*'${bs1_one:0:5}'}'`
echo "bs1_fps: ${bs1_fps}"

bs16_one=`cat result/perf_vision_batchsize_16_device_0.txt | grep 'Interface throughputRate' | awk '{print $6}'`
bs16_fps=`awk 'BEGIN{printf "%.2f\n", 4*'${bs16_one:0:5}'}'`
echo "bs16_fps: ${bs16_fps}"


















