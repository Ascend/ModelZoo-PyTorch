rm -rf TextSnake.onnx
python TextSnake_pth2onnx.py --input_file './textsnake_vgg_180.pth'  --output_file './TextSnake.onnx'

source env.sh
rm -rf TextSnake_bs1.om
atc --model=TextSnake.onnx --framework=5 --output=TextSnake_bs1 --input_format=NCHW --input_shape="image:1,3,512,512" --log=info --soc_version=Ascend310
rm -rf TextSnake_bs16.om
atc --model=TextSnake.onnx --framework=5 --output=TextSnake_bs16 --input_format=NCHW --input_shape="image:16,3,512,512" --log=info --soc_version=Ascend310

if [ -f "TextSnake_bs1.om" ] && [ -f "TextSnake_bs16.om" ]; then
    echo "success"
else
    echo "fail!"
fi
