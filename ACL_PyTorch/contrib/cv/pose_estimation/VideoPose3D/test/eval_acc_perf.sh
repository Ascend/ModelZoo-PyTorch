#!/bin/bash

datasets_path="./VideoPose3D/data"

for para in $*
do
    if [[ $para == --datasets_path* ]]; then
        datasets_path=`echo ${para#*=}`
    fi
done

# arch=`uname -m`

echo "Preprocessing data ..."

rm -rf ./preprocessed_data
python vp3d_preprocess.py -d ${datasets_path}

if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

source /usr/local/Ascend/ascend-toolkit/set_env.sh

./msame --model "vp3d_seq6115.om" --input "./preprocessed_data/inputs" --output "./preprocessed_data/outputs" --outfmt BIN > inference.log

if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

outfile=$(ls preprocessed_data/outputs/)

echo "Computing accuracy..."
python vp3d_postprocess.py -o ${outfile}

if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

echo "==== Performance Data ===="
acc=`grep -a "average" inference.log|awk 'END {print}'|awk -F " " '{print $7 / 4}'`
echo "Average Inference time:$acc ms"

echo "success"
