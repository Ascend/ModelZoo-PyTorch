source /usr/local/Ascend/ascend-toolkit/set_env.sh
rm -rf shufflenetv2_bs1.onnx
rm -rf shufflenetv2_bs16.onnx
python3.7 shufflenetv2_pth2onnx_bs1.py ShuffleNetV2+.Small.pth.tar shufflenetv2_bs1.onnx
python3.7 shufflenetv2_pth2onnx_bs16.py ShuffleNetV2+.Small.pth.tar shufflenetv2_bs16.onnx
rm -rf shufflenetv2_bs1.om shufflenetv2_bs16.om
atc --framework=5 --model=./shufflenetv2_bs1.onnx --input_format=NCHW --input_shape="image:1,3,224,224" --output=shufflenetv2_bs1 --log=debug --soc_version=Ascend310
atc --framework=5 --model=./shufflenetv2_bs16.onnx --input_format=NCHW --input_shape="image:16,3,224,224" --output=shufflenetv2_bs16 --log=debug --soc_version=Ascend310
if [ -f "shufflenetv2_bs1.om" ] && [ -f "shufflenetv2_bs16.om" ]; then
	echo "success"
else
	echo "fail!"
fi
