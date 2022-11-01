#!/bin/bash

unset ASCEND_SLOG_PRINT_TO_STDOUT

python tools/i3d_inference.py configs/recognition/i3d/i3d_nl_dot_product_r50_32x2x1_100e_kinetics400_rgb.py --eval top_k_accuracy mean_class_accuracy --out result.json -bs 1 --model i3d_nl_dot_bs1.om --device_id 3 
