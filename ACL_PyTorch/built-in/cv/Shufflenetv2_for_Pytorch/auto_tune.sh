source /usr/local/Ascend/ascend-toolkit/set_env.sh
# export REPEAT_TUNE=True

atc --model=$1 --framework=5 --output=./shufflenetV2 --input_format=NCHW --input_shape="actual_input_1:1,3,224,224" --log=info --soc_version=Ascend310 --auto_tune_mode="RL,GA"