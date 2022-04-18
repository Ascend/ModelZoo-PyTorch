#!/bin/bash

set -eu

datasets_path="data/coco"
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

echo "==== preprocess ===="
rm -rf val2017_bin
python uniformer_preprocess.py --dataset=${datasets_path}

if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

source /usr/local/Ascend/ascend-toolkit/set_env.sh

echo "==== inference ===="
rm -rf result
if [ $batch_size == 1 ]; then
    ./msame --model=uniformer_bs1.om --input=val2017_bin --output=result

    if [ $? != 0 ]; then
        echo "fail!"
        exit -1
    fi
else
    rm -rf val2017_bin_packed result_packed
    python test/bin_util.py --pack --input=val2017_bin --output=val2017_bin_packed --batch_size=$batch_size
    ./msame --model=uniformer_bs$batch_size.om --input=val2017_bin_packed --output=result_packed
    python test/bin_util.py --unpack --input=result_packed --output=result --batch_size=$batch_size

    if [ $? != 0 ]; then
        echo "fail!"
        exit -1
    fi
fi

echo "==== postprocess ===="
rm -rf tmp
python uniformer_postprocess.py --dataset=${datasets_path}

if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

echo "success"
