dims=""
for i in $(seq 96); do
  let "token_length=$i+1"
  dims=$dims$token_length','$i';'
done
echo $dims

atc --model=m2m_decoder.onnx \
    --framework=5 \
    --output=m2m_decoder \
    --input_format=ND \
    --precesion_mode=allow_mix_precision \
    --modify_mixlist=ops_info.json \
    --input_fp16_nodes="65;66" \
    --output_type=FP16 \
    --input_shape="prev_output_tokens:5,-1;65:120,16,-1,64" \
    --dynamic_dims=$dims \
    --log=error \
    --soc_version=Ascend310P3