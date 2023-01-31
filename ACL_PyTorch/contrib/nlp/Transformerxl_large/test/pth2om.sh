python3 transformerxl_large_pth2onnx.py --work_dir=${1} --batch_size=${2}
python3 -m onnxsim model_bs${2}.onnx model_bs${2}_sim.onnx --input-shape "data:128,${2}" "target:128,${2}"
python3 fix_int64.py model_bs${2}_sim.onnx model_bs${2}_sim_fix.onnx
source env.sh
atc --framework=5 --model=model_bs${2}_sim_fix.onnx --output=model_bs${2}  --input_format=ND --input_shape="data:128,${2};target:128,${2}" --log=debug --soc_version=Ascend310 --fusion_switch_file=fusion_switch.cfg