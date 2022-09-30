# 配置环境变量 
source /usr/local/Ascend/ascend-toolkit/set_env.sh
 
# 使用二进制输入时，执行如下命令。不开启aipp，用于精度测试
${install_path}/atc/bin/atc --model=./pelee_dynamic_bs_modify.onnx --framework=5 --output=pelee_bs1 --input_format=NCHW --input_shape="image:1,3,304,304" --log=info --soc_version=$1 --enable_small_channel=1
 
# 使用二进制输入时，执行如下命令。开启aipp，用于性能测试 
${install_path}/atc/bin/atc --model=./pelee_dynamic_bs_modify.onnx --framework=5 --output=pelee_bs32 --input_format=NCHW --input_shape="image:32,3,304,304" --log=info --soc_version=$1 --enable_small_channel=1 --insert_op_conf=aipp.config 