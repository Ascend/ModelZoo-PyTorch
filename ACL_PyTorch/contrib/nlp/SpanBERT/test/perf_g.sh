#!/bin/bash

bs=1

for para in $*
do
    if [[ $para == --bs* ]]; then
        bs=`echo ${para#*=}`
    fi
done
python ./test/perf_t4.py --bs ${bs}