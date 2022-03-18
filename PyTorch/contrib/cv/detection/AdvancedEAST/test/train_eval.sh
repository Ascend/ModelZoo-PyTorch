#!/bin/bash

source test/env.sh

if [ -n "$*" ]
then
    SIZES=$*
else
    SIZES="736"
fi

for SIZE in $SIZES
do
    PTH_PATH 
    nohup python3.7 eval.py --pth_path saved_model/3T${SIZE}_latest.pth --size $SIZE
    sleep 5s
done
