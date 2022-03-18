#! /bin/bash
export install_path=/usr/local/Ascend/ascend-toolkit/latest
export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export PYTHONPATH=${install_path}/atc/python/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=${install_path}/atc/lib64:${install_path}/acllib/lib64:$LD_LIBRARY_PATH
export ASCEND_OPP_PATH=${install_path}/opp
export ASCEND_AICPU_PATH=/usr/local/Ascend/ascend-toolkit/latest

# start
python LResNet_pth2onnx.py ./model/model_ir_se100.pth ./model/model_ir_se100_bs1.onnx 1;
python LResNet_pth2onnx.py ./model/model_ir_se100.pth ./model/model_ir_se100_bs16.onnx 16;

python -m onnxsim --input-shape="1,3,112,112" ./model/model_ir_se100_bs1.onnx ./model/model_ir_se100_bs1_sim.onnx;
python -m onnxsim --input-shape="16,3,112,112" ./model/model_ir_se100_bs16.onnx ./model/model_ir_se100_bs16_sim.onnx;

atc --framework=5 --model=./model/model_ir_se100_bs1_sim.onnx --output=model/model_ir_se100_bs1 --input_format=NCHW --input_shape="image:1,3,112,112" --log=debug --soc_version=Ascend310;
atc --framework=5 --model=./model/model_ir_se100_bs16_sim.onnx --output=model/model_ir_se100_bs16 --input_format=NCHW --input_shape="image:16,3,112,112" --log=debug --soc_version=Ascend310;