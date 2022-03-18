bash tar2onnx.sh
atc --framework=5 --model='bmn-bs1.onnx' --output='bmn-bs1' --input_format=NCHW --input_shape="image:1,400,100" --log=debug --soc_version=Ascend310