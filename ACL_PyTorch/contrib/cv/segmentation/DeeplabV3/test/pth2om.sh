path=`pwd`
rm -rf deeplabv3.onnx deeplabv3_sim_bs1.onnx
python ${path}/mmsegmentation/tools/pytorch2onnx.py \
${path}/mmsegmentation/configs/deeplabv3/deeplabv3_r50-d8_512x1024_40k_cityscapes.py \
--checkpoint ${path}/deeplabv3_r50-d8_512x1024_40k_cityscapes_20200605_022449-acadc2f8.pth \
--output-file ${path}/deeplabv3.onnx --shape 1024 2048
python -m onnxsim deeplabv3.onnx deeplabv3_sim_bs1.onnx --input-shape="1,3,1024,2048" --dynamic-input-shape
source /usr/local/Ascend/ascend-toolkit/set_env.sh
rm -rf deeplabv3_bs1.om
atc --framework=5 --model=${path}/deeplabv3_sim_bs1.onnx --output=${path}/deeplabv3_bs1 --input_format=NCHW  --input_shape="input:1,3,1024,2048" --log=debug --soc_version=Ascend310 --auto_tune_mode="RL,GA"
if [ -f "deeplabv3_bs1.om" ] ; then
    echo "success"
else
    echo "fail!"
fi

