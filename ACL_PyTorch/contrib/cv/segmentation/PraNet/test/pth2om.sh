python3.7 PraNet_pth2onnx.py   ./PraNet-19.pth  ./PraNet-19.onnx
python3.7 -m onnxsim PraNet-19.onnx PraNet-19_dybs_sim.onnx --input-shape=1,3,352,352 --dynamic-input-shape

atc --framework=5 --model=PraNet-19_dybs_sim.onnx --output=PraNet-19_bs1 --input_format=NCHW --input_shape="actual_input_1:1,3,352,352"  --log=debug --soc_version=Ascend310

atc --framework=5 --model=PraNet-19_dybs_sim.onnx --output=PraNet-19_bs16 --input_format=NCHW --input_shape="actual_input_1:16,3,352,352" --log=debug --soc_version=Ascend310

