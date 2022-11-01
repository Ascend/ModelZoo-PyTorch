#!/bin/bash

python tools/data/build_rawframes.py data/kinetics400/videos_val data/kinetics400/rawframes_val --task rgb --level 1 --num-worker 4 --out-format jpg --ext mp4 --new-width 256 --new-height 256 --use-opencv