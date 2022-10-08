source /usr/local/Ascend/ascend-toolkit/set_env.sh
export SLOG_PRINT_TO_STDOUT=1
# export REPEAT_TUNE=True

atc --model=$1 --framework=5 --output=$2 --input_format=NCHW --input_shape="actual_input_1:16,3,224,224" --enable_small_channel=1 --log=info --soc_version=Ascend310

# --input_fp16_nodes="actual_input_1"
# --enable_small_channel=1
# --auto_tune_mode="RL,GA"