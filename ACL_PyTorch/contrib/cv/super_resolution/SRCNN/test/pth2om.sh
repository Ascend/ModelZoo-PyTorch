rm -rf *.onnx
python3.7 srcnn_pth2onnx.py --pth srcnn_x2.pth --onnx srcnn_x2.onnx
source env.sh
rm -rf *.om
atc --framework=5 --model=srcnn_x2.onnx --output=srcnn_x2 --input_format=NCHW --input_shape="input.1:1,1,256,256" --log=debug --soc_version=Ascend310
if [ -f "srcnn_x2.om" ]; then
    echo "Success changing pth to om."
else
    echo "Fail!"
fi