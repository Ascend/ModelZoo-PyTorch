rm -rf *.onnx
python3.7 rcan_pth2onnx.py --pth RCAN_BIX2.pt --onnx rcan.onnx
source /usr/local/Ascend/ascend-toolkit/set_env.sh
rm -rf *.om
atc --framework=5 --model=rcan.onnx --output=rcan_1bs --input_format=NCHW --input_shape="image:1,3,256,256" --fusion_switch_file=switch.cfg --log=debug --soc_version=Ascend310
if [ -f "rcan_1bs.om" ]; then
    echo "Success changing pth to om."
else
    echo "Fail!"
fi
