rm -rf *.onnx
python3.7 srflow_pth2onnx.py  --pth ./SRFlow/pretrained_models/SRFlow_DF2K_8X.pth --onnx srflow_df2k_x8.onnx
python3.7 -m onnxsim srflow_df2k_x8.onnx srflow_df2k_x8_sim.onnx 0 --input-shape "1,3,256,256"
source env.sh
rm -rf *.om
atc --framework=5 --model=srflow_df2k_x8_sim.onnx --output=srflow_df2k_x8_bs1 --input_format=NCHW --input_shape="input_1:1,3,256,256" --log=debug --soc_version=Ascend310 --fusion_switch_file=./fusion_switch.cfg
if [ -f "srflow_df2k_x8_bs1.om" ]; then
    echo "Success changing pth to om."
else
    echo "Fail!"
fi