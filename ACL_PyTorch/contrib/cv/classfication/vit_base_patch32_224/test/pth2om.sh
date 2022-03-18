python3.7 vit_base_patch32_224_pth2onnx.py --batch-size 1
python3.7 vit_base_patch32_224_pth2onnx.py --batch-size 16
source env.sh
python3.7 -m onnxsim ./vit_bs1.onnx ./vit_bs1_sim.onnx --input-shape "input:1,3,224,224"
python3.7 -m onnxsim ./vit_bs16.onnx ./vit_bs16_sim.onnx --input-shape "input:16,3,224,224"
atc --framework=5 --model=vit_bs1_sim.onnx --output=vit_bs1 --input_format=NCHW --input_shape="input:1,3,224,224" --log=debug --soc_version=Ascend310 --precision_mode=allow_mix_precision --modify_mixlist=ops_info.json
atc --framework=5 --model=vit_bs16_sim.onnx --output=vit_bs16 --input_format=NCHW --input_shape="input:16,3,224,224" --log=debug --soc_version=Ascend310 --precision_mode=allow_mix_precision --modify_mixlist=ops_info.json
if [ -f "vit_bs1.om" ] && [ -f "vit_bs16.om" ]; then
    echo "success"
else
    echo "fail!"
fi