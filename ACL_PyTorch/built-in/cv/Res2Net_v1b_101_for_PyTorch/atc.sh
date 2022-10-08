# 配置环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 310P fp16，执行如下命令
atc --model=./res2net.onnx \
    --framework=5 \
    --output=res2net_bs16 \
    --input_format=NCHW \
    --input_shape="x:16,3,224,224" \
    --log=error \
    --soc_version=$1 \
    --enable_small_channel=1
