#!/bin/bash

python3 ./tools/deployment/pytorch2onnx.py \
		./configs/recognition/r2plus1d/r2plus1d_ucf101_rgb_1p.py ./work_dirs/r2plus1d-1p-npu/best_top1_acc_epoch_35.pth \
		--verify  --output-file=r2plus1d.onnx --shape 1 3 3 8 256 256
		
#简化onnx。
python3 -m onnxsim --input-shape="1,3,3,8,256,256" --dynamic-input-shape r2plus1d.onnx r2plus1d_sim.onnx
		
	