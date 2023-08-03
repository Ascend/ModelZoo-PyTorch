atc --model=ctc_dynamic.onnx --framework=5 --output=ctc --input_format=ND \
--input_shape="x:-1,-1,256" --log=error \
--input_fp16_nodes="x" --output_type="FP16" --soc_version=$1
