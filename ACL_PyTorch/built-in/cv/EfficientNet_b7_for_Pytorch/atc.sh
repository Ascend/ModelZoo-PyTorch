source /usr/local/Ascend/toolbox/set_env.sh
soc_version=`ascend-dmi -i -dt | grep -m 1 'Chip Name' | awk -F ': ' '{print $2}' | sed 's/ //g'`

source /usr/local/Ascend/ascend-toolkit/set_env.sh
onnx_model=$1
output_model=$2
atc --model=$onnx_model \
    --framework=5 \
    --input_format=NCHW \
    --input_shape="image:32,3,600,600" \
    --output=$output_model \
    --soc_version=${soc_version} \
    --log=error \
    --optypelist_for_implmode="Sigmoid" \
    --op_select_implmode=high_performance 