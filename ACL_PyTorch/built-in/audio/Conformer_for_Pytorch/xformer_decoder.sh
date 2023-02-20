atc --model=xformer_decoder_revise.onnx --framework=5 --output=xformer_decoder --input_format=ND \
--input_shape_range="tgt:[-1,-1];memory:[-1,-1,256];cache_0:[-1,-1,256];cache_1:[-1,-1,256];cache_2:[-1,-1,256]; \
cache_3:[-1,-1,256];cache_4:[-1,-1,256];cache_5:[-1,-1,256]" --log=error \
--soc_version=$1
