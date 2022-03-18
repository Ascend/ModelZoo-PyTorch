#!/bin/bash

python tools/deployment/pytorch2onnx.py configs/recognition/i3d/i3d_nl_dot_product_r50_32x2x1_100e_kinetics400_rgb.py checkpoints/i3d_nl_dot_product_r50.pth --shape 1 10 3 32 256 256 --output-file i3d_nl_dot.onnx