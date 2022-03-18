if [ $4 == Ascend310 ];then
    atc --model=$1 \
        --framework=5 \
        --output=$2_bs$3 \
        --input_format=NCHW \
        --input_shape="images:$3,3,640,640;img_info:$3,4" \
        --log=error \
        --soc_version=$4 \
        --input_fp16_nodes="images;img_info" \
        --output_type=FP16
fi

if [ $4 == Ascend710 ];then
    atc --model=$1 \
        --framework=5 \
        --output=$2_bs$3 \
        --input_format=NCHW \
        --input_shape="images:$3,3,640,640;img_info:$3,4" \
        --log=error \
        --soc_version=$4 \
        --input_fp16_nodes="images;img_info" \
        --output_type=FP16 \
        --optypelist_for_implmode="Sigmoid" \
        --op_select_implmode=high_performance
fi
