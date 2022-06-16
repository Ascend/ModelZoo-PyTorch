#!/bin/bash

python3 edsr_preprocess.py -s benchmark/B100/LR_bicubic/X2/ -d data_preprocessed/B100
python3 dynamic_infer.py
python3 edsr_postprocess.py --res outputs/B100 --HR benchmark/B100/HR --save_path .
