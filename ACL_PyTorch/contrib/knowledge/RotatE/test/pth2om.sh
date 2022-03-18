echo "====onnx===="

rm -rf kge_onnx_head.onnx
rm -rf kge_onnx_tail.onnx

python3.7 rotate_pth2onnx.py --pth_path="./checkpoint"   --onnx_path="./kge_onnx_head.onnx"  --mode="head-batch" 
python3.7 rotate_pth2onnx.py --pth_path="./checkpoint"   --onnx_path="./kge_onnx_tail.onnx"  --mode="tail-batch" 

echo "====om transform begin===="
source env.sh
rm -rf kge_1_head.om
rm -rf kge_1_tail.om
rm -rf kge_16_head.om
rm -rf kge_16_tail.om
atc --framework=5 --model=kge_onnx_head.onnx --output=kge_1_head --input_format=ND --input_shape="pos:1,3;neg:1,14541" --log=error --soc_version=Ascend310 
atc --framework=5 --model=kge_onnx_tail.onnx --output=kge_1_tail --input_format=ND --input_shape="pos:1,3;neg:1,14541" --log=error --soc_version=Ascend310 
atc --framework=5 --model=kge_onnx_head.onnx --output=kge_16_head --input_format=ND --input_shape="pos:16,3;neg:16,14541" --log=error --soc_version=Ascend310 
atc --framework=5 --model=kge_onnx_tail.onnx --output=kge_16_tail --input_format=ND --input_shape="pos:16,3;neg:16,14541" --log=error --soc_version=Ascend310 


if [ -f "kge_1_head.om" ] && [ -f "kge_1_tail.om" ] && [ -f "kge_16_head.om" ] && [ -f "kge_16_tail.om" ]; then
    echo "success"
else
    echo "fail!"
fi
