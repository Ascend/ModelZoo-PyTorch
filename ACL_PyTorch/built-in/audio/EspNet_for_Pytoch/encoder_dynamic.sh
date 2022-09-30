atc --model=encoder_revise.onnx --framework=5 --output=encoder --input_format=ND \
--input_shape_range="input:[1~1500,83]" --log=error --op_select_implmode=high_performance \
--soc_version=$1
