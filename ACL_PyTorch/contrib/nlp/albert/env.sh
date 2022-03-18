# encoding=utf-8

source /usr/local/Ascend/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH=/usr/local/python3.7.5/lib/python3.7/site-packages/torch/lib:${LD_LIBRARY_PATH}  #shm的so文件
export LD_LIBRARY_PATH=/usr/local/python3.7.5/lib:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/:${LD_LIBRARY_PATH}

source /usr/local/Ascend/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH=/usr/local/python3.7.5/lib/python3.7/site-packages/torch/lib:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=/usr/local/python3.7.5/lib:${LD_LIBRARY_PATH}

export install_path=/usr/local/Ascend/ascend-toolkit/latest
export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export PYTHONPATH=${install_path}/atc/python/site-packages:$PYTHONPATH
export ASCEND_OPP_PATH=${install_path}/opp
export ASCEND_AICPU_PATH=/usr/local/Ascend/ascend-toolkit/latest

export REPEAT_TUNE=True

# 开启TASK多线程下发
export TASK_QUEUE_ENABLE=1
# 用于导出host日志到屏幕
export ASCEND_SLOG_PRINT_TO_STDOUT=1
# 日志级别设置，信息从多到少分别是 debug 0 --> info 1 --> warning 2 --> error 3 --> null，一般设置为error，调试时使用info，--device用于设置对应哪张卡
export ASCEND_GLOBAL_LOG_LEVEL=3
# event 0关1开
export ASCEND_GLOBAL_EVENT_ENABLE=0
# 开启AICPU PTCOPY (用于缓解某些场景下的非连续转连续问题)
export PTCOPY_ENABLE=1
export COMBINED_ENABLE=1