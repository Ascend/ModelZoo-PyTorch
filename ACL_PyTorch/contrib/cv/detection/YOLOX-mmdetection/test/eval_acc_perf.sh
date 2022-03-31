#!/bin/bash

set -eu

datasets_path="/root/datasets"
batch_size=1

for para in $*
do
    if [[ $para == --datasets_path* ]]; then
        datasets_path=`echo ${para#*=}`
    fi
    if [[ $para == --batch_size* ]]; then
        batch_size=`echo ${para#*=}`
    fi
done


arch=`uname -m`

rm -rf val2017_bin
rm -rf val2017_bin_meta
python YOLOX_preprocess.py --image_src_path ${datasets_path}/coco/val2017

if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

python gen_dataset_info.py \
${datasets_path}/coco \
mmdetection/configs/yolox/yolox_s_8x8_300e_coco.py  \
val2017_bin  val2017_bin_meta  \
yolox.info  yolox_meta.info  \
640 640

if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

source /usr/local/Ascend/ascend-toolkit/set_env.sh
rm -rf result

./benchmark.${arch} -model_type=vision -om_path=yolox.om -device_id=0 -batch_size=${batch_size} \
-input_text_path=yolox.info -input_width=640 -input_height=640 -useDvpp=false -output_binary=true

if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

python YOLOX_postprocess.py --dataset_path ${datasets_path}/coco
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi


echo "====performance data===="
python test/parse.py result/perf_vision_batchsize_${batch_size}_device_0.txt

if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "success"

