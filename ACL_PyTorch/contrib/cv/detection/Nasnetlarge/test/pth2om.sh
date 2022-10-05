source /usr/local/Ascend/ascend-toolkit/set_env.sh
rm -rf nasnetlarge.onnx

python3.7 nasnetlarge_pth2onnx.py nasnetalarge-a1897284.pth nasnetlarge.onnx

python3.7 -m onnxsim --input-shape="1,3,331,331" nasnetlarge.onnx nasnetlarge_sim_bs1.onnx
python3.7 -m onnxsim --input-shape="16,3,331,331" nasnetlarge.onnx nasnetlarge_sim_bs16.onnx

python merge_sliced.py nasnetlarge_sim_bs1.onnx nasnetlarge_sim_merge_bs1.onnx
python merge_sliced.py nasnetlarge_sim_bs16.onnx nasnetlarge_sim_merge_bs16.onnx

rm -rf nasnetlarge_bs1.om nasnetlarge_bs16.om
atc --framework=5 --model=nasnetlarge_sim_merge_bs1.onnx --input_format=NCHW --input_shape="image:1,3,331,331" --output=nasnetlarge_sim_bs1 --log=debug --soc_version=Ascend310 --auto_tune_mode="RL,GA" 
atc --framework=5 --model=nasnetlarge_sim_merge_bs16.onnx --input_format=NCHW --input_shape="image:16,3,331,331" --output=nasnetlarge_sim_bs16 --log=debug --soc_version=Ascend310 --auto_tune_mode="RL,GA" 

if [ -f "nasnetlarge_sim_bs1.om" ] && [ -f "nasnetlarge_sim_bs16.om" ]; then
	echo "success"
else
	echo "fail!"
fi
