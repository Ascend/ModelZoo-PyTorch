source /usr/local/Ascend/ascend-toolkit/set_env.sh

atc --model=$1 --framework=5 --output=./mobileNetV2_1_npu_autoTune --input_format=NCHW --input_shape="actual_input_1:1,3,224,224" --log=info --soc_version=Ascend310 --auto_tune_mode="RL,GA"