echo "====onnx===="

rm -rf fusion_result.json
rm -rf kernel_meta
rm -rf dqn.onnx

python3.7 pth2onnx.py --pth-path='dqn.pth' --onnx-path='dqn.onnx'
if [ -f "dqn.onnx" ] ;  then
    echo "success"
else
    echo "fail!"
fi

