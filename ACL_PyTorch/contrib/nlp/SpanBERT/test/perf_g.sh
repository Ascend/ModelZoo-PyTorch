#!/bin/bash

bs=1
path=./spanBert_dynamicbs.onnx 

for para in $*
do
    if [[ $para == --bs* ]]; then
        bs=`echo ${para#*=}`
    fi
    if [[ $para == --path* ]]; then
        path=`echo ${para#*=}`
    fi
done

python ./test/perf_t4.py --bs ${bs} --path ${path}