atc --model=xformer_decoder_revise.onnx --framework=5 --output=xformer_decoder --input_format=ND \
--input_shape="tgt:-1,-1;memory:-1,-1,256;cache_0:-1,-1,256;cache_1:-1,-1,256;cache_2:-1,-1,256; \
cache_3:-1,-1,256;cache_4:-1,-1,256;cache_5:-1,-1,256" --log=error --output_type="FP16" \
--input_fp16_nodes="memory;cache_0;cache_1;cache_2;cache_3;cache_4;cache_5" --soc_version=$1
