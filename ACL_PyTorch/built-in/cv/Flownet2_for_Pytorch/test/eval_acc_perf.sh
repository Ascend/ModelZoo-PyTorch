#!/bin/bash

# bs1
python3.7 preprocess.py --batch_size 1 --dataset ./MPI-Sintel-complete/training --output ./data_preprocessed_bs1

./msame --model models/flownet2_bs1_sim_fix.om --input data_preprocessed_bs1/image1/,data_preprocessed_bs1/image2/ --output output_bs1/

python3.7 evaluate.py --gt_path ./data_preprocessed_bs1/gt --output_path ./output_bs1/ --batch_size 1

