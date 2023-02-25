atc --model=xformer_encoder.onnx --framework=5 --output=xformer_encoder --input_format=ND \
--input_shape_range="feats:[1,-1,80]" --log=error \
--soc_version=$1
