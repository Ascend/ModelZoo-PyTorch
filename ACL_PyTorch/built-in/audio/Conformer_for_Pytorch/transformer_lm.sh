atc --model=transformer_lm_revise.onnx --framework=5 --output=transformer_lm --input_format=ND \
--input_shape_range="tgt:[-1,-1];cache_0:[-1,-1,512];cache_1:[-1,-1,512];cache_2:[-1,-1,512]; \
cache_3:[-1,-1,512];cache_4:[-1,-1,512];cache_5:[-1,-1,512];cache_6:[-1,-1,512];cache_7:[-1,-1,512]; \
cache_8:[-1,-1,512];cache_9:[-1,-1,512];cache_10:[-1,-1,512];cache_11:[-1,-1,512];cache_12:[-1,-1,512]; \
cache_13:[-1,-1,512];cache_14:[-1,-1,512];cache_15:[-1,-1,512]" --log=error \
--soc_version=$1
