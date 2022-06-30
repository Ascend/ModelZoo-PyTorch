# pytorch to onnx
wget -P ./mmsegmentation/  https://download.openmmlab.com/mmsegmentation/v0.5/stdc/stdc1_512x1024_80k_cityscapes/stdc1_512x1024_80k_cityscapes_20220224_073048-74e6920a.pth

cd mmsegmentation
python tools/pytorch2onnx.py \
    configs/stdc/stdc1_512x1024_80k_cityscapes.py \
    --checkpoint stdc1_512x1024_80k_cityscapes_20220224_073048-74e6920a.pth \
    --output-file ../stdc_bs1.onnx \
    --cfg-options model.test_cfg.mode="whole"

if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

# optimize onnx
cd ..
python optimize_onnx.py stdc_bs1.onnx stdc_optimize_bs1.onnx

if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

# onnx to om
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# STDC bs1
atc --framework=5 --model=./stdc_optimize_bs1.onnx --output=stdc_optimize_bs1 --input_format=NCHW --input_shape="input:1,3,1024,2048" --log=debug --soc_version=$1 --insert_op_conf=./aipp.config --enable_small_channel=1

if [ -f "stdc_optimize_bs1.om" ]; then
    echo "success"
else
    echo "fail!"
fi
