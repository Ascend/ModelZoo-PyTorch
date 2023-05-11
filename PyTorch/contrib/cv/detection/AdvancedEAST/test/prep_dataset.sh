#!/bin/bash

SIZES="256 384 512 640 736"

for SIZE in $SIZES
do
    python3 preprocess.py --size $SIZE
    python3 label.py --size $SIZE
done
