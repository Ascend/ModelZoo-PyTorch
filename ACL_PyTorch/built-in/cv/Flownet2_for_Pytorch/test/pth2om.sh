#!/bin/bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# convert onnx
python3.7 pth2onnx.py --batch_size 1 --input_path ./FlowNet2_checkpoint.pth.tar --out_path ./models/flownet2_bs1.onnx --batch_size 1

# bs1
# optimize onnx
python3.7 -m onnxsim ./models/flownet2_bs1.onnx ./models/flownet2_bs1_sim.onnx
python3.7 fix_onnx.py ./models/flownet2_bs1_sim.onnx ./models/flownet2_bs1_sim_fix.onnx
# 310需要采用混合精度，否则有精度问题；310P上采用FP16精度正常
# atc --framework=5 --model=models/flownet2_bs1_sim_fix.onnx --output=models/flownet2_bs1_sim_fix --input_format=NCHW --input_shape="x1:1,3,448,1024;x2:1,3,448,1024" --log=debug --soc_version=Ascend310 --precision_mode=allow_mix_precision
atc --framework=5 --model=models/flownet2_bs1_sim_fix.onnx --output=models/flownet2_bs1_sim_fix --input_format=NCHW --input_shape="x1:1,3,448,1024;x2:1,3,448,1024" --log=debug --soc_version=$1
