#CANN安装目录
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export install_path=/usr/local/Ascend/ascend-toolkit/latest
export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export PYTHONPATH=${install_path}/atc/python/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=${install_path}/atc/lib64:${install_path}/acllib/lib64:$LD_LIBRARY_PATH
export ASCEND_OPP_PATH=${install_path}/opp
export ASCEND_AICPU_PATH=/usr/local/Ascend/ascend-toolkit/latest
export LD_LIBRARY_PATH=/usr/local/python3.7.5/lib/python3.7/site-packages/torch/lib:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=/usr/local/python3.7.5/lib:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/:${LD_LIBRARY_PATH}

#将atc日志打印到屏�?#export ASCEND_SLOG_PRINT_TO_STDOUT=1
#设置日志级别
export ASCEND_GLOBAL_LOG_LEVEL=2 #debug 0 --> info 1 --> warning 2 --> error 3
#开启ge dump�?    #开启这个中间文件太多了
#参考命�?#framework: 0(Caffe) or 1(MindSpore) or 3(TensorFlow) or 5(Onnx).
atc --framework=5 --model=onnxmodel/model_Ga.onnx --output=model_Ga-b0_bs1 --input_format=NCHW --input_shape="img_sat_maps:1,3,256,256" --out_nodes="Tanh_156:0" --log=debug --soc_version=Ascend310
atc --framework=5 --model=onnxmodel/model_Gb.onnx --output=model_Gb-b0_bs1 --input_format=NCHW --input_shape="img_maps_sat:1,3,256,256"  --out_nodes="Tanh_156:0" --log=debug --soc_version=Ascend310

atc --framework=5 --model=onnxmodel/model_Ga.onnx --output=model_Ga-b0_bs16 --input_format=NCHW --input_shape="img_sat_maps:16,3,256,256" --out_nodes="Tanh_156:0" --log=debug --soc_version=Ascend310
atc --framework=5 --model=onnxmodel/model_Gb.onnx --output=model_Gb-b0_bs16 --input_format=NCHW --input_shape="img_maps_sat:16,3,256,256"  --out_nodes="Tanh_156:0" --log=debug --soc_version=Ascend310