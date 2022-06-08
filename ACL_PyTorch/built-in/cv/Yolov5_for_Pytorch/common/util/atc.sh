mode=$1
onnx=$2
om=$3
bs=$4
soc=$5


if [ ${mode} == val ];then
    input_shape="images:${bs},3,640,640"
    input_fp16_nodes="images"
elif [ ${mode} == infer ];then
    input_shape="images:${bs},3,640,640;img_info:${bs},4"
    input_fp16_nodes="images;img_info"
fi

if [ ${soc} == Ascend310 ];then
    atc --model=${onnx} \
        --framework=5 \
        --output=${om}_bs${bs} \
        --input_format=NCHW \
        --input_shape=${input_shape} \
        --log=error \
        --soc_version=${soc} \
        --input_fp16_nodes=${input_fp16_nodes} \
        --output_type=FP16
fi

if [ ${soc} == Ascend710 ];then
    atc --model=${onnx} \
        --framework=5 \
        --output=${om}_bs${bs} \
        --input_format=NCHW \
        --input_shape=${input_shape} \
        --log=error \
        --soc_version=${soc} \
        --input_fp16_nodes=${input_fp16_nodes} \
        --output_type=FP16 \
        --optypelist_for_implmode="Sigmoid" \
        --op_select_implmode=high_performance \
        --fusion_switch_file=common/util/fusion.cfg
fi
