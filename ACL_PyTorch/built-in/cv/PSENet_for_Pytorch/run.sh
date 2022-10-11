source /usr/local/Ascend/ascend-toolkit/set_env.sh

atc \
--model=$1 \
--framework=5 \
--output=$2 \
--input_format=NCHW --input_shape="actual_input_1:1,3,704,1216" \
--enable_small_channel=1 \
--log=error \
--soc_version=Ascend310 \
--insert_op_conf=aipp.config