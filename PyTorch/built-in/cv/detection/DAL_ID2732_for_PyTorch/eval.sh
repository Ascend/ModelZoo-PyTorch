#!/usr/bin/env bash
source env.sh

python3 main_eval.py \
    --backbone res101 \
    --dataset UCAS_AOD \
    --target_size "800,1344" \
    --weight weights/result_last.pth \
    --root_path datasets/evaluate \
    --test_path ${UCAS_AOD_PATH}/test.txt \
    --device_index 0