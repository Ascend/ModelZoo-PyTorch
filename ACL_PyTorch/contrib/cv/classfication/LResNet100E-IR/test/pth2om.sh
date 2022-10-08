#! /bin/bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# start
python LResNet_pth2onnx.py ./model/model_ir_se100.pth ./model/model_ir_se100_bs1.onnx 1;
python LResNet_pth2onnx.py ./model/model_ir_se100.pth ./model/model_ir_se100_bs16.onnx 16;

python -m onnxsim --input-shape="1,3,112,112" ./model/model_ir_se100_bs1.onnx ./model/model_ir_se100_bs1_sim.onnx;
python -m onnxsim --input-shape="16,3,112,112" ./model/model_ir_se100_bs16.onnx ./model/model_ir_se100_bs16_sim.onnx;

atc --framework=5 --model=./model/model_ir_se100_bs1_sim.onnx --output=model/model_ir_se100_bs1 --input_format=NCHW --input_shape="image:1,3,112,112" --log=debug --soc_version=Ascend310;
atc --framework=5 --model=./model/model_ir_se100_bs16_sim.onnx --output=model/model_ir_se100_bs16 --input_format=NCHW --input_shape="image:16,3,112,112" --log=debug --soc_version=Ascend310;
