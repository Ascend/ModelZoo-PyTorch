source /usr/local/Ascend/ascend-toolkit/set_env.sh
rm -rf dino_resnet50.onnx
python3.7 dino_resnet50_pth2onnx.py
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

rm -rf dino_resnet50_bs1.om dino_resnet50_bs16.om

# ${chip_name}可通过 npu-smi info指令查看
atc --framework=5 --model=dino_resnet50.onnx --output=dino_resnet50_bs1 --input_format=NCHW --input_shape="input:1,3,224,224" --log=debug --soc_version=Ascend${chip_name}
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
atc --framework=5 --model=dino_resnet50.onnx --output=dino_resnet50_bs16 --input_format=NCHW --input_shape="input:16,3,224,224" --log=debug --soc_version=Ascend${chip_name}
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
