python3.7 CGAN_pth2onnx.py --pth_path CGAN_G.pth --onnx_path CGAN.onnx
python3.7 -m onnxsim --input-shape="100,72" CGAN.onnx CGAN_sim.onnx
source /usr/local/Ascend/ascend-toolkit/set_env.sh
atc --framework=5 --model=CGAN_sim.onnx --output=CGAN_bs1 --output_type=FP32 --input_format=ND --input_shape="image:100,72" --log=debug --soc_version=Ascend310 --auto_tune_mode="RL,GA"
