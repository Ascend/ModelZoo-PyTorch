echo "====onnx===="

rm -rf fusion_result.json
rm -rf kernel_meta
rm -rf c51.onnx

python3.7 pth2onnx.py --model-path='c51.model' --onnx-path='c51.onnx'
if [ -f "c51.onnx" ] ;  then
    echo "success"
else
    echo "fail!"
fi

echo "====om transform begin===="
source env.sh
rm -rf c51_bs1.om
atc --framework=5 --model=c51.onnx --output=c51_bs1 --input_format=NCHW --input_shape="input:1,4,84,84" --auto_tune_mode="RL,GA" --log=error --soc_version=Ascend310  --op_select_implmode=high_performance

if [ -f "c51_bs1.om" ];  then
    echo "success"
else
    echo "fail!"
fi
