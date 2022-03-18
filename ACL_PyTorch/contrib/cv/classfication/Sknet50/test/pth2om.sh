rm -rf *.onnx
python3.7 sknet2onnx.py --pth sk_resnet50.pth.tar --onnx sknet50.onnx
source env.sh
rm -rf *.om
atc --framework=5 --model=sknet50.onnx --output=sknet50_1bs --input_format=NCHW --input_shape="image:1,3,224,224" --log=debug --soc_version=Ascend310
atc --framework=5 --model=sknet50.onnx --output=sknet50_16bs --input_format=NCHW --input_shape="image:16,3,224,224" --log=debug --soc_version=Ascend310
if [ -f "sknet50_1bs.om" ] && [ -f "sknet50_16bs.om" ]; then
    echo "Success changing pth to om."
else
    echo "Fail!"
fi