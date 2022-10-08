echo "====onnx===="

rm -rf r2plus1d.onnx
python3.7 ./mmaction2/tools/deployment/pytorch2onnx.py ./mmaction2/configs/recognition/r2plus1d/r2plus1d_r34_8x8x1_180e_ucf101_rgb2.py best_top1_acc_epoch_35.pth --verify  --output-file=r2plus1d.onnx --shape 1 3 3 8 256 256

python3.7 -m onnxsim --input-shape="1,3,3,8,256,256" --dynamic-input-shape r2plus1d.onnx r2plus1d_sim.onnx

source /usr/local/Ascend/ascend-toolkit/set_env.sh
rm -rf r2plus1d_bs1.om
rm -rf r2plus1d_bs16.om
atc --framework=5 --model=./r2plus1d_sim.onnx --output=r2plus1d_bs1 --input_format=NCHW --input_shape="0:1,3,3,8,256,256" --log=debug --soc_version=Ascend310

atc --framework=5 --model=./r2plus1d_sim.onnx --output=r2plus1d_bs16 --input_format=NCHW --input_shape="0:16,3,3,8,256,256" --log=debug --soc_version=Ascend310
if [ -f "r2plus1d_bs1.om" ] && [ -f "r2plus1d_bs16.om" ]; then
    echo "success"
else
    echo "fail!"
fi
