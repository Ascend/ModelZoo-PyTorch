# 配置环境变量
export install_path=/usr/local/Ascend/ascend-toolkit/latest
export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export PYTHONPATH=${install_path}/atc/python/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=${install_path}/atc/lib64:${install_path}/acllib/lib64:$LD_LIBRARY_PATH
export ASCEND_OPP_PATH=${install_path}/opp

# 710 fp16，执行如下命令
atc --model=./res2net.onnx \
    --framework=5 \
    --output=res2net_bs16 \
    --input_format=NCHW \
    --input_shape="x:16,3,224,224" \
    --log=error \
    --soc_version=Ascend710 \
    --enable_small_channel=1
