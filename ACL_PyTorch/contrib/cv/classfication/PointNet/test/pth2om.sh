source /usr/local/Ascend/ascend-toolkit/set_env.sh
python3.7 pointnet_pth2onnx.py
python3.7 -m onnxsim pointnet.onnx pointnet_bs1_sim.onnx --input-shape="1, 3, 2500" --dynamic-input-shape
python3.7 fix_conv1d.py pointnet_bs1_sim.onnx pointnet_bs1_sim_fixed.onnx
atc --framework=5 --model=pointnet_bs1_sim_fixed.onnx --output=pointnet_bs1_fixed --input_shape="image:1, 3, 2500" --soc_version=Ascend310 --log=error > atc1.log

python3.7 -m onnxsim pointnet.onnx pointnet_bs4_sim.onnx --input-shape="4, 3, 2500" --dynamic-input-shape
python3.7 fix_conv1d.py pointnet_bs4_sim.onnx pointnet_bs4_sim_fixed.onnx
atc --framework=5 --model=pointnet_bs4_sim_fixed.onnx --output=pointnet_bs4_fixed --input_shape="image:4, 3, 2500" --soc_version=Ascend310 --log=error > atc4.log

python3.7 -m onnxsim pointnet.onnx pointnet_bs8_sim.onnx --input-shape="8, 3, 2500" --dynamic-input-shape
python3.7 fix_conv1d.py pointnet_bs8_sim.onnx pointnet_bs8_sim_fixed.onnx
atc --framework=5 --model=pointnet_bs8_sim_fixed.onnx --output=pointnet_bs8_fixed --input_shape="image:8, 3, 2500" --soc_version=Ascend310 --log=error > atc8.log

python3.7 -m onnxsim pointnet.onnx pointnet_bs16_sim.onnx --input-shape="16, 3, 2500" --dynamic-input-shape
python3.7 fix_conv1d.py pointnet_bs16_sim.onnx pointnet_bs16_sim_fixed.onnx
atc --framework=5 --model=pointnet_bs16_sim_fixed.onnx --output=pointnet_bs16_fixed --input_shape="image:16, 3, 2500" --soc_version=Ascend310 --log=error > atc16.log

python3.7 -m onnxsim pointnet.onnx pointnet_bs32_sim.onnx --input-shape="32, 3, 2500" --dynamic-input-shape
python3.7 fix_conv1d.py pointnet_bs32_sim.onnx pointnet_bs32_sim_fixed.onnx
atc --framework=5 --model=pointnet_bs32_sim_fixed.onnx --output=pointnet_bs32_fixed --input_shape="image:32, 3, 2500" --soc_version=Ascend310 --log=error > atc32.log

