#!/bin/bash

source /usr/local/Ascend/ascend-toolkit/set_env.sh

datasets_path="/root/datasets/"

for para in $*
do
    if [[ $para == --datasets_path* ]]; then
        datasets_path=`echo ${para#*=}`
    fi
done

arch=`uname -m`
# rm -rf ${datasets_path}/coco/prep_dataset
python3.7 OpenPose_preprocess.py --src_path ${datasets_path}/coco/val2017 --save_path ${datasets_path}/coco/prep_dataset --pad_txt_path ./output/pad.txt
if [ $? != 0 ]; then
    echo "preprocess fail!"
    exit -1
fi
python3.7 gen_dataset_info.py bin ../../../../../root/datasets/coco/prep_dataset ./output/openpose_prep_bin.info 640 368
if [ $? != 0 ]; then
    echo "info fail!"
    exit -1
fi

# rm -rf result/dumpOutput_device0
./benchmark.${arch} -model_type=vision -device_id=0 -batch_size=1 -om_path=./output/human-pose-estimation_bs1.om -input_text_path=./output/openpose_prep_bin.info -input_width=640 -input_height=368 -output_binary=False -useDvpp=False
if [ $? != 0 ]; then
    echo "benchmark bs1 fail!"
    exit -1
fi
# rm -rf result/dumpOutput_device1
./benchmark.${arch} -model_type=vision -device_id=1 -batch_size=16 -om_path=./output/human-pose-estimation_bs16.om -input_text_path=./output/openpose_prep_bin.info -input_width=640 -input_height=368 -output_binary=False -useDvpp=False
if [ $? != 0 ]; then
    echo "benchmark bs16 fail!"
    exit -1
fi

echo "====accuracy data===="
python3.7 OpenPose_postprocess.py --benchmark_result_path result/dumpOutput_device0/ --labels ${datasets_path}/coco/annotations/person_keypoints_val2017.json --pad_txt_path ./output/pad.txt --detections_save_path ./output/result_b1.json
if [ $? != 0 ]; then
    echo "postprocess bs1 fail!"
    exit -1
fi
python3.7 OpenPose_postprocess.py --benchmark_result_path result/dumpOutput_device1/ --labels ${datasets_path}/coco/annotations/person_keypoints_val2017.json --pad_txt_path ./output/pad.txt --detections_save_path ./output/result_b16.json
if [ $? != 0 ]; then
    echo "postprocess bs16 fail!"
    exit -1
fi

echo "====performance data===="
python3.7 test/parse.py result/perf_vision_batchsize_1_device_0.txt
if [ $? != 0 ]; then
    echo "parse fail!"
    exit -1
fi
python3.7 test/parse.py result/perf_vision_batchsize_16_device_1.txt
if [ $? != 0 ]; then
    echo "parse fail!"
    exit -1
fi
echo "success"
