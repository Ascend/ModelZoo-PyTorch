model_path=$1
framework=$2
output_model_name=$3

/usr/local/Ascend/atc/bin/atc \
--model=$model_path \
--framework=$framework \
--output=$output_model_name \
--input_format=NCHW \
--input_shape="actual_input_1:1,3,304,304" \
--enable_small_channel=1 \
--disable_reuse_memory=1 \
--buffer_optimize=off_optimize \
--log=error \
--soc_version=Ascend310 \
--insert_op_conf=./aipp.config
