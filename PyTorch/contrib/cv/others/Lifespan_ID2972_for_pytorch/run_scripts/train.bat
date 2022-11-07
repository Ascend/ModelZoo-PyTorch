@echo off

set CUDA_VISIBLE_DEVICES=0,1,2,3

python train.py --npu_ids 0,1,2,3 --dataroot ./datasets/males --name males_model --batchSize 6 --verbose
