#!/bin/bash
dataset_path="/root/datasets"

for para in $*
do
    if [[ $para == --dataset_path* ]]; then
        dataset_path=`echo ${para#*=}`
    fi
done

valid_images=$dataset_path/images
valid_info=$dataset_path/annotations/instances_val2017.json

arch=`uname -m`
benchmark_path=./benchmark.$arch

rm yolact_prep_bin.info
rm -rf ./prep_dataset
rm -rf ./result
rm -rf ./kernel_meta
rm fusion_result.json
rm -rf ./results

echo "Begin data preprocessing!"

python3.7.5 YOLACT_preprocess.py --valid_images=$valid_images --valid_annotations=$valid_info --config=yolact_plus_resnet50_config

echo "perform offline eval!"

source /usr/local/Ascend/ascend-toolkit/set_env.sh

$benchmark_path -model_type=vision -device_id=0 -batch_size=1 -om_path=yolact_plus_bs1.om -input_text_path=./yolact_prep_bin.info -input_width=550 -input_height=550 -output_binary=True -useDvpp=False

$benchmark_path -model_type=vision -device_id=1 -batch_size=8 -om_path=yolact_plus_bs8.om -input_text_path=./yolact_prep_bin.info -input_width=550 -input_height=550 -output_binary=True -useDvpp=False

python3.7.5 YOLACT_postprocess.py --valid_images=$valid_images --valid_annotations=$valid_info --config=yolact_plus_resnet50_config --device_id=0

python3.7.5 YOLACT_postprocess.py --valid_images=$valid_images --valid_annotations=$valid_info --config=yolact_plus_resnet50_config --device_id=1

python3.7.5 parse.py result/perf_vision_batchsize_$2_device_0.txt

python3.7.5 parse.py result/perf_vision_batchsize_$2_device_1.txt