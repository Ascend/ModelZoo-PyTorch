

python RawNet2_pth2onnx.py -name x


python3.7 -m onnxsim --input-shape="1,59049" RawNet2.onnx RawNet2_sim_bs1.onnx

python3.7 -m onnxsim --input-shape="16,59049" RawNet2.onnx RawNet2_sim_bs16.onnx

source env.sh

atc --framework=5 --model=RawNet2_sim_bs1.onnx --output=RawNet2_sim_bs1 --input_format=ND --input_shape="wav:1,59049" --log=error --soc_version=Ascend310   --fusion_switch_file=fusion_switch.cfg

atc --framework=5 --model=RawNet2_sim_bs16.onnx --output=RawNet2_sim_bs16 --input_format=ND --input_shape="wav:16,59049" --log=error --soc_version=Ascend310   --fusion_switch_file=fusion_switch.cfg