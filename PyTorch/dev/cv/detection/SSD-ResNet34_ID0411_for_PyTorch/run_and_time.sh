#!/bin/bash
# runs benchmark and reports time to convergence
# to use the script:
#   run_and_time.sh <random seed> <target threshold>
#--checkpoint ./models/iter_120000.pt --iteration 120000
SEED=${1:-1}
TARGET=${2:-0.2812}
DATASET_DIR='../coco'

time stdbuf -o 0 \
  python3 train.py --seed $SEED --threshold $TARGET --data ${DATASET_DIR} --device 1  | tee run.log.$SEED 
