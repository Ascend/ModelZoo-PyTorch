source /usr/local/Ascend/ascend-toolkit/set_env.sh
# export REPEAT_TUNE=True

atc --model=$1 --framework=5 --output=$2 --input_format=NCHW --input_shape="actual_input_1:16,3,224,224" --enable_small_channel=1 --log=info --soc_version=Ascend310