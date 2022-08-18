soc=$1
output_dir=$2
model=$3
bs=$4


atc --model=${output_dir}/${model}.onnx \
    --framework=5 \
    --output=${output_dir}/${model}_bs${bs} \
    --input_format=NCHW \
    --input_shape="input:${bs},3,224,224" \
    --log=error \
    --soc_version=${soc} \
    --input_fp16_nodes="input" \
    --output_type=FP16
