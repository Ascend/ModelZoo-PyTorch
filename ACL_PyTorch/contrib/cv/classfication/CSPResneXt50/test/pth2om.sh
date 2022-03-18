source env.sh
rm -rf cspresnext.onnx
python3.7 cspresnext_pth2onnx.py cspresnext50_ra_224-648b4713.pth cspresnext.onnx
rm -rf cspresnext_bs1.om cspresnext_bs16.om
atc --framework=5 --model=./cspresnext.onnx --input_format=NCHW --input_shape="image:1,3,224,224" --output=cspresnext_bs1 --log=debug --soc_version=Ascend310
atc --framework=5 --model=./cspresnext.onnx --input_format=NCHW --input_shape="image:16,3,224,224" --output=cspresnext_bs16 --log=debug --soc_version=Ascend310
if [ -f "cspresnext_bs1.om" ] && [ -f "cspresnext_bs16.om" ]; then
	echo "success"
else
	echo "fail!"
fi
