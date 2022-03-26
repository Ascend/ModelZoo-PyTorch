atc --model=no_flash_encoder_revise.onnx --framework=5 --output=encoder_fendang_262_1478_static --input_format=ND \
--input_shape="xs_input:1,-1,80;xs_input_lens:1" --log=error \
--dynamic_dims="262;326;390;454;518;582;646;710;774;838;902;966;1028;1284;1478" \
--soc_version=Ascend310
