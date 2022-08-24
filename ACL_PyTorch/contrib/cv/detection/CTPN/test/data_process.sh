#!/bin/bash

datasets_path="./data/Challenge2_Test_Task12_Images"

for para in $*
do
    if [[ $para == --datasets_path* ]]; then
        datasets_path=`echo ${para#*=}`
    fi
done

arch=`uname -m`
python3.7 task_process.py --mode='preprocess' --src_dir=${datasets_path}
if [ $? != 0 ]; then
    echo "preprocess fail!"
    exit -1
fi