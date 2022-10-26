# 配置环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh

export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_GLOBAL_EVENT_ENABLE=0
export ASCEND_SLOG_PRINT_TO_STDOUT=0
/usr/local/Ascend/driver/tools/msnpureport -g error -d 0
/usr/local/Ascend/driver/tools/msnpureport -g error -d 1
/usr/local/Ascend/driver/tools/msnpureport -g error -d 2
# 310P fp16，执行如下命令
atc --model=./resnet34_dynamic.onnx --framework=5 --output=resnet34_fp16_bs8 --input_format=NCHW --input_shape="actual_input_1:8,3,224,224" --log=info --soc_version=$1 --insert_op_conf=resnet34_aipp.config

# 310 fp16，执行如下命令
# atc --model=./resnet34_dynamic.onnx --framework=5 --output=resnet34_fp16_bs16 --input_format=NCHW --input_shape="actual_input_1:16,3,224,224" --log=info --soc_version=Ascend310 --insert_op_conf=resnet34_aipp.config

# 310P int8，执行如下命令
atc --model=./resnet34_deploy_model.onnx --framework=5 --output=resnet34_int8_bs16 --input_format=NCHW --input_shape="actual_input_1:16,3,224,224" --log=info --soc_version=$1 --insert_op_conf=resnet34_aipp.config

# 310 int8，执行如下命令
# atc --model=./resnet34_deploy_model.onnx --framework=5 --output=resnet34_int8_bs16 --input_format=NCHW --input_shape="actual_input_1:16,3,224,224" --log=info --soc_version=Ascend310 --insert_op_conf=resnet34_aipp.config
