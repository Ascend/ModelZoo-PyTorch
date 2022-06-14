onnx_model=$1
output_model=$2
soc_version=$3

atc --model=$onnx_model \
    --framework=5 \
    --input_format=NCHW \
    --input_shape="image:32,3,600,600" \
    --output=$output_model \
    --soc_version=${soc_version} \
    --log=error \
    --optypelist_for_implmode="Sigmoid" \
    --op_select_implmode=high_performance 