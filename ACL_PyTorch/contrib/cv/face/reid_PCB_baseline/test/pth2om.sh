#! /bin/bash
echo "--------------------------------------------"
python3.7 ../pth2onnx.py
echo "--------------------------------------------"
python3.7 -m onnxsim --input-shape="1,3,384,128" ./models/PCB.onnx  ./models/PCB_sim_bs1.onnx
echo "--------------------------------------------"
python3.7 -m onnxsim --input-shape="16,3,384,128" ./models/PCB.onnx  ./models/PCB_sim_bs16.onnx
#bs1
echo "--------------------------------------------"
python3.7 ./scripts/split_reducelp.py ./models/PCB_sim_bs1.onnx ./models/PCB_sim_split_bs1.onnx
#bs16
echo "--------------------------------------------"
python3.7 ./scripts/split_reducelp.py ./models/PCB_sim_bs16.onnx ./models/PCB_sim_split_bs16.onnx
#转OM模型 bs=1
echo "--------------------------------------------"
atc --framework=5 --model=./models/PCB_sim_split_bs1.onnx --output=./models/PCB_sim_split_bs1_autotune --input_format=NCHW --input_shape="input_1:1,3,384,128" --log=debug --soc_version=Ascend310 --auto_tune_mode="RL,GA" --out_nodes='Div_126:0;Gemm_191:0;Gemm_192:0;Gemm_193:0;Gemm_194:0;Gemm_195:0;Gemm_196:0'
#转OM模型 bs=16
echo "--------------------------------------------"
atc --framework=5 --model=./models/PCB_sim_split_bs16.onnx --output=./models/PCB_sim_split_bs16_autotune --input_format=NCHW --input_shape="input_1:16,3,384,128" --log=debug --soc_version=Ascend310 --auto_tune_mode="RL,GA" --out_nodes='Div_126:0;Gemm_191:0;Gemm_192:0;Gemm_193:0;Gemm_194:0;Gemm_195:0;Gemm_196:0'
