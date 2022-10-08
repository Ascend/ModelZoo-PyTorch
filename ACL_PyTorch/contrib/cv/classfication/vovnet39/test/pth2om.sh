source /usr/local/Ascend/ascend-toolkit/set_env.sh
rm -rf vovnet39.onnx
python3.7 vovnet39_pth2onnx.py vovnet39_torchvision.pth vovnet39.onnx
rm -rf vovnet39_bs1.om vovnet39_bs16.om
atc --framework=5 --model=./vovnet39.onnx --input_format=NCHW --input_shape="image:1,3,224,224" --output=vovnet39_bs1 --log=debug --soc_version=Ascend310
atc --framework=5 --model=./vovnet39.onnx --input_format=NCHW --input_shape="image:16,3,224,224" --output=vovnet39_bs16 --log=debug --soc_version=Ascend310
if [ -f "vovnet39_bs1.om" ] && [ -f "vovnet39_bs16.om" ]; then
	echo "success"
else
	echo "fail!"
fi
