source /usr/local/Ascend/ascend-toolkit/set_env.sh
# export REPEAT_TUNE=True
 
/usr/local/Ascend/ascend-toolkit/latest/atc/bin/atc \
--model=$1 \
--framework=5 \
--output=$2 \
--input_format=NCHW \
--input_shape="actual_input_1:1,3,704,1216" \
--enable_small_channel=1 \
--log=info \
--soc_version=Ascend310 \
--auto_tune_mode="RL,GA"