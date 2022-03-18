python pth2onnx.py --config SOLO/configs/solov2/solov2_r50_fpn_8gpu_1x.py --pth_path SOLOv2_R50_1x.pth --out SOLOv2.onnx --shape 800 1216
python -m onnxsim SOLOv2.onnx SOLOv2_sim.onnx
source env.sh
atc --framework=5 --model=SOLOv2_sim.onnx --output=solov2  --input_format=NCHW --input_shape="input:1,3,800,1216" --log=error --soc_version=Ascend310

if [ -f "solov2.om" ]; then
    echo "success"
else
    echo "fail!"
fi