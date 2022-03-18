python pth2onnx.py --config SOLO/configs/solo/solo_r50_fpn_8gpu_1x.py --pth_path SOLO_R50_1x.pth --out SOLOv1.onnx --shape 800 1216
python -m onnxsim SOLOv1.onnx SOLOv1_sim.onnx
source env.sh
atc --framework=5 --model=SOLOv1_sim.onnx --output=solo  --input_format=NCHW --input_shape="input:1,3,800,1216" --log=error --soc_version=Ascend310

if [ -f "solo.om" ]; then
    echo "success"
else
    echo "fail!"
fi