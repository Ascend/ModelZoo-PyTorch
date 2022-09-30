atc --model=offline_encoder.onnx --framework=5 --output=encoder_static --input_format=ND \
--input_shape="speech:32,-1,80;speech_lengths:32" --log=error \
--dynamic_dims="262;326;390;454;518;582;646;710;774;838;902;966;1028;1284;1478" \
--soc_version=$1
