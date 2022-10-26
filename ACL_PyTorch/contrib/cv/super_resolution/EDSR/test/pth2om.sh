rm -rf *.onnx
python3.7 edsr_pth2onnx.py --pth edsr_x2.pt --onnx edsr_x2.onnx --size 1020
source /usr/local/Ascend/ascend-toolkit/set_env.sh
rm -rf *.om
atc --framework=5 --model=edsr_x2.onnx --output=edsr_x2 --input_format=NCHW --input_shape="input.1:1,3,1020,1020" --log=debug --soc_version=Ascend310 --fusion_switch_file=switch.cfg
if [ -f "edsr_x2.om" ]; then
    echo "Success changing pth to om."
else
    echo "Fail!"
fi
