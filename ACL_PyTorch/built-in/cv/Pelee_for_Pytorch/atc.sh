# 配置环境变量 
export install_path=/usr/local/Ascend/ascend-toolkit/latest 
export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH 
export PYTHONPATH=${install_path}/atc/python/site-packages:$PYTHONPATH 
export LD_LIBRARY_PATH=${install_path}/atc/lib64:${install_path}/acllib/lib64:$LD_LIBRARY_PATH 
export ASCEND_OPP_PATH=${install_path}/opp 
 
# 使用二进制输入时，执行如下命令。不开启aipp，用于精度测试
${install_path}/atc/bin/atc --model=./pelee_dynamic_bs_modify.onnx --framework=5 --output=pelee_bs1 --input_format=NCHW --input_shape="image:1,3,304,304" --log=info --soc_version=Ascend710 --enable_small_channel=1
 
# 使用二进制输入时，执行如下命令。开启aipp，用于性能测试 
${install_path}/atc/bin/atc --model=./pelee_dynamic_bs_modify.onnx --framework=5 --output=pelee_bs32 --input_format=NCHW --input_shape="image:32,3,304,304" --log=info --soc_version=Ascend710 --enable_small_channel=1 --insert_op_conf=aipp.config 