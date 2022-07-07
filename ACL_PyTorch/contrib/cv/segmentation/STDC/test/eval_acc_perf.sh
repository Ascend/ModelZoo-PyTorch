#!/bin/bash

source /usr/local/Ascend/ascend-toolkit/set_env.sh

# preprocess data
python ./STDC_preprocess.py $1/cityscapes/leftImg8bit/val/ ./prep_dataset

if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

# inference with msame
rm -rf ./output

./msame --model=./stdc_optimize_bs1.om --input=./prep_dataset --output=./output --outfmt=BIN

if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

# postprocess data
python ./STDC_postprocess.py --output_path=./output/$(ls output) --gt_path=$1/cityscapes/gtFine/val

if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

echo "success"