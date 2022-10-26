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
python data_preprocess.py --image_src_path=${datasets_path}/coco/val2017 \
--bin_file_path=val2017_bin --model_input_height=800 --model_input_width=1216

if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

python get_info.py jpg ${datasets_path}/coco/val2017 swin_jpeg.info

if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

source /usr/local/Ascend/ascend-toolkit/set_env.sh
rm -rf result

./msame --model swin.om --input val2017_bin --output result --outfmt bin > ./msame_bs${batch_size}.txt


if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

python -u data_postprocess.py --bin_data_path=./result \
--test_annotation=swin_jpeg.info  \
--net_out_num=3 --net_input_height=800 --net_input_width=1216 --ifShowDetObj \
--anno_path ${datasets_path}/coco/annotations/instances_val2017.json --data_path ${datasets_path}/coco/val2017

if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

# =========================== print performance data =========================
echo "====performance data===="
python test/parse.py --result-file ./msame_bs${batch_size}.txt --batch-size ${batch_size}
if [ $? != 0 ]; then
    echo "parse bs${batch_size} performance fail!"
    exit -1
fi