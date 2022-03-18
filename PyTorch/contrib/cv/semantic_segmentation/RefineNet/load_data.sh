#!/bin/bash

python3 src/load_dataset.py \
    --enc-backbone 101 \
    --num-stages 2 \
    --num-classes 21 \
    --train-dir './VOC' \
    --val-dir './VOC' \
    --dataset-type 'torchvision' \
    --stage-names 'SBD' 'VOC' \
    --epochs-per-stage 100 100 \
    --augmentations-type 'albumentations' \
    --train-download 1 1 \
    --val-download 1
