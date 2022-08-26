echo "====onnx===="

source /usr/local/Ascend/ascend-toolkit/set_env.sh

rm -rf fusion_result.json
rm -rf kernel_meta
rm -rf ssd_bs1.onnx
rm -rf ssd_bs16.onnx

python ssd_pth2onnx.py --bs=1 --resnet34-model=./models/resnet34-333f7ec4.pth --pth-path=./models/iter_183250.pt --onnx-path=./ssd_bs1.onnx

if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

python ssd_pth2onnx.py --bs=16 --resnet34-model=./models/resnet34-333f7ec4.pth --pth-path=./models/iter_183250.pt --onnx-path=./ssd_bs16.onnx

if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

rm -rf ssd_bs1.om
rm -rf ssd_bs16.om

echo "====om transform begin===="
#这个环境变量和上面的环境变量不一样
export install_path=/usr/local/Ascend/ascend-toolkit/latest
export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export PYTHONPATH=${install_path}/atc/python/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=${install_path}/atc/lib64:${install_path}/acllib/lib64:$LD_LIBRARY_PATH
export ASCEND_OPP_PATH=${install_path}/opp
export ASCEND_AICPU_PATH=/usr/local/Ascend/ascend-toolkit/latest

atc --framework=5 --model=./ssd_bs1.onnx --output=./ssd_bs1 --input_format=NCHW --input_shape="image:1,3,300,300" --log=error --soc_version=Ascend310

if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

atc --framework=5 --model=./ssd_bs16.onnx --output=./ssd_bs16 --input_format=NCHW --input_shape="image:16,3,300,300" --log=error --soc_version=Ascend310

if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

echo "success"
