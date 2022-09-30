atc --model=offline_encoder.onnx --framework=5 --output=offline_encoder --input_format=ND --input_shape_range="speech:[1~64,1~1500,80];speech_lengths:[1~64]" --log=error  --soc_version=$1

