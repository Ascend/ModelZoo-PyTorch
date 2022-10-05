#!/bin/bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
echo "====TEM pth2onnx===="

python BSN_tem_pth2onnx.py --pth_path './tem_best.pth.tar' --onnx_path './BSN_tem.onnx'
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "====TEM pth2onnx finished===="

echo "====TEM onnx conv1d2conv2d===="
python TEM_onnx_conv1d2conv2d.py './BSN_tem.onnx' './BSN_tem1.onnx'
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "====TEM onnx conv1d2conv2d finished===="

echo "====PEM pth2onnx===="

python BSN_pem_pth2onnx.py --pth_path './pem_best.pth.tar' --onnx_path './BSN_pem.onnx'
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "====TPM pth2onnx finished===="

install_path=/usr/local/Ascend/ascend-toolkit/latest
export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export PYTHONPATH=${install_path}/atc/python/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=${install_path}/atc/lib64:${install_path}/acllib/lib64:$LD_LIBRARY_PATH
export ASCEND_OPP_PATH=${install_path}/opp
export ASCEND_AICPU_PATH=/usr/local/Ascend/ascend-toolkit/latest
export ASCEND_SLOG_PRINT_TO_STDOUT=1

echo "====TEM onnx2om bs1===="

atc --framework=5 --model=BSN_tem1.onnx --output=BSN_tem_bs1 --input_format=ND --input_shape="video:1,400,100" --log=debug --soc_version=Ascend310
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "====TPM onnx2om bs1 finished===="
echo "====TEM onnx2om bs16===="

atc --framework=5 --model=BSN_tem1.onnx --output=BSN_tem_bs16 --input_format=ND--input_shape="video:16,400,100" --log=debug --soc_version=Ascend310
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "====TPM onnx2om bs16 finished===="

echo "====PEM onnx2om bs1===="

atc --framework=5 --model=BSN_pem.onnx --output=BSN_pem_bs1 --input_format=ND --input_shape="video_feature:1,1000,32" --log=debug --soc_version=Ascend310
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "====PPM onnx2om bs1 finished===="
echo "====PEM onnx2om bs16===="

atc --framework=5 --model=BSN_pem.onnx --output=BSN_pem_bs16 --input_format=ND --input_shape="video_feature:16,1000,32" --log=debug --soc_version=Ascend310
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "====PEM onnx2om bs16 finished===="
