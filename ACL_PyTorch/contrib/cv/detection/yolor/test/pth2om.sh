rm -rf yolor_bs1.onnx
python yolor_pth2onnx.py --cfg ./yolor_p6_swish.cfg --weights ./yolor_p6.pt --output_file ./yolor_bs1.onnx --batch_size 1
rm -rf yolor_bs1_sim.onnx
python -m onnxsim --input-shape='1,3,1344,1344' yolor_bs1.onnx yolor_bs1_sim.onnx

source env.sh
rm -rf yolor_bs1.om
atc --model=yolor_bs1_sim.onnx --framework=5 --output=yolor_bs1 --input_format=NCHW --input_shape="image:1,3,1344,1344" --log=info --soc_version=Ascend310 --out_nodes="Concat_2575:0" --buffer_optimize=off_optimize

if [ -f "yolor_bs1.om" ]; then
    echo "success"
else
    echo "fail!"
fi