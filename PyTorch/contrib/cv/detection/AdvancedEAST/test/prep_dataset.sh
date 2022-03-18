#!/bin/bash

SIZES="256 384 512 640 736"

for SIZE in $SIZES
do
    python3.7 preprocess.py --size $SIZE
    python3.7 label.py --size $SIZE
done
