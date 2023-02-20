atc --model=ctc.onnx --framework=5 --output=ctc --input_format=ND \
--input_shape_range="x:[1,-1,256]" --log=error \
--soc_version=$1
