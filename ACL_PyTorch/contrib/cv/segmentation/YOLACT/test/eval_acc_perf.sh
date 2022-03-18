#!/bin/bash
valid_images=$1
valid_info=$2
batch_size=$3
om_path=$4
benchmark_path=$5

if [ $# -ne 5 ]
then
	echo "param num is $#, but 5 params is needed!"
	echo "format : ./eval_acc_perf.sh [valid_images] [valid_info] [batch_size] [om_path] [benchmark_path]"
	exit
else
	echo "valid_images:$1, valid_info:$2, batch_size:$3, om_path:$4, benchmark_path:$5"
fi

rm yolact_prep_bin.info
rm -rf ./prep_dataset
rm -rf ./result
rm -rf ./kernel_meta
rm fusion_result.json
rm -rf ./results

echo "Begin data preprocessing!"

python ../YOLACT_preprocess.py --valid_images=$valid_images --valid_annotations=$valid_info

echo "perform offline eval!"

source ../env.sh

$5 -model_type=vision -device_id=0 -batch_size=$3 -om_path=$4 -input_text_path=./yolact_prep_bin.info -input_width=550 -input_height=550 -output_binary=True -useDvpp=False

echo "====================================accuracy data===================================="

python ../YOLACT_postprocess.py --valid_images=$1 --valid_annotations=$2

echo "==================================performance data==================================="
python ./parse.py ./result/perf_vision_batchsize_$3_device_0.txt

echo "success!"
