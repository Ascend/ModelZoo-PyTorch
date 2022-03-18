#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python3 eval.py experiments/seg_detector/ic15_resnet50_deform_thre.yaml --resume path-to-model-directory/ic15_resnet50 --box_thresh 0.6 --speed
CUDA_VISIBLE_DEVICES=0 python3 eval.py experiments/seg_detector/ic15_resnet50_deform_thre.yaml --resume path-to-model-directory/ic15_resnet50 --box_thresh 0.6 --batchsize 16
