source /usr/local/Ascend/ascend-toolkit/set_env.sh
rm -rf shufflenetv1_bs1.onnx
rm -rf shufflenetv1_bs16.onnx
python3.7 shufflenetv1_pth2onnx_bs1.py 1.0x.pth.tar shufflenetv1_bs1.onnx
python3.7 shufflenetv1_pth2onnx_bs16.py 1.0x.pth.tar shufflenetv1_bs16.onnx
rm -rf shufflenetv1_bs1.om shufflenetv1_bs16.om
atc --framework=5 --model=./shufflenetv1_bs1.onnx --input_format=NCHW --input_shape="image:1,3,224,224" --output=shufflenetv1_bs1 --log=debug --soc_version=Ascend710
atc --framework=5 --model=./shufflenetv1_bs16.onnx --input_format=NCHW --input_shape="image:16,3,224,224" --output=shufflenetv1_bs16 --log=debug --soc_version=Ascend710
if [ -f "shufflenetv1_bs1.om" ] && [ -f "shufflenetv1_bs16.om" ]; then
	echo "success"
else
	echo "fail!"
fi
