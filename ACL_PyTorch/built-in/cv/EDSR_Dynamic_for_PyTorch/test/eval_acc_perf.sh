#!/bin/bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PYTHONPATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:$PYTHONPATH

python3 edsr_preprocess.py -s benchmark/B100/LR_bicubic/X2/ -d data_preprocessed/B100
python3 dynamic_infer.py -m models/om/EDSR_x2.om -s ./outputs/B100
python3 edsr_postprocess.py --res outputs/B100 --HR benchmark/B100/HR --save_path .