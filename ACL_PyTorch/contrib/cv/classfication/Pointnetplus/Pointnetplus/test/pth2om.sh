#!/bin/bash

set -eu
# set -x

source env.sh

python pointnetplus_pth2onnx.py --target_model 1 --pth_dir './models/log/classification/pointnet2_ssg_wo_normals/checkpoints' --batch_size 1
python pointnetplus_pth2onnx.py --target_model 2 --pth_dir './models/log/classification/pointnet2_ssg_wo_normals/checkpoints' --batch_size 1
python pointnetplus_pth2onnx.py --target_model 1 --pth_dir './models/log/classification/pointnet2_ssg_wo_normals/checkpoints' --batch_size 16
python pointnetplus_pth2onnx.py --target_model 2 --pth_dir './models/log/classification/pointnet2_ssg_wo_normals/checkpoints' --batch_size 16

atc --framework=5 --model=Pointnetplus_part1_bs1.onnx --output=Pointnetplus_part1_bs1 --input_format=ND --input_shape="samp_points:1,3,32,512" --log=debug --soc_version=Ascend310
atc --framework=5 --model=Pointnetplus_part2_bs1.onnx --output=Pointnetplus_part2_bs1 --input_format=ND --input_shape="l1_points:1,131,64,128;l1_xyz:1,3,128" --log=debug --soc_version=Ascend310
atc --framework=5 --model=Pointnetplus_part1_bs16.onnx --output=Pointnetplus_part1_bs16 --input_format=ND --input_shape="samp_points:16,3,32,512" --log=debug --soc_version=Ascend310
atc --framework=5 --model=Pointnetplus_part2_bs16.onnx --output=Pointnetplus_part2_bs16 --input_format=ND --input_shape="l1_points:16,131,64,128;l1_xyz:16,3,128" --log=debug --soc_version=Ascend310
