#!/bin/bash

source test/env.sh

if [ -n "$*" ]
then
    SIZES=$*
else
    SIZES="256 384 512 640 736"
fi

for SIZE in $SIZES
do
    nohup python3.7 -u train.py --size $SIZE --apex --epoch_num 3 --val_interval 1
    sleep 5s
done
