export install_path=/usr/local/Ascend/ascend-toolkit/latest
export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export PYTHONPATH=${install_path}/atc/python/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=${install_path}/atc/lib64:${install_path}/acllib/lib64:$LD_LIBRARY_PATH
export ASCEND_OPP_PATH=${install_path}/opp
export ASCEND_AICPU_PATH=/usr/local/Ascend/ascend-toolkit/latest
# 将atc日志打印到屏幕
# export ASCEND_SLOG_PRINT_TO_STDOUT=1
# 设置日志级别
export ASCEND_GLOBAL_LOG_LEVEL=0 #debug 0 --> info 1 --> warning 2 --> error 3
#开启ge dump图
#export DUMP_GE_GRAPH=2
