# ############## nnae situation ################
# if [ -d /usr/local/Ascend/nnae/latest ];then
#   export LD_LIBRARY_PATH=/usr/local/:/usr/local/python3.7.5/lib/:/usr/local/openblas/lib:/usr/local/lib/:/usr/lib64/:/usr/lib/:/usr/local/Ascend/nnae/latest/fwkacllib/lib64/:/usr/local/Ascend/driver/lib64/common/:/usr/local/Ascend/driver/lib64/driver/:/usr/local/Ascend/add-ons/:/usr/lib/aarch64_64-linux-gnu:$LD_LIBRARY_PATH
#   export PATH=$PATH:/usr/local/Ascend/nnae/latest/fwkacllib/ccec_compiler/bin/:/usr/local/Ascend/nnae/latest/toolkit/tools/ide_daemon/bin/
#   export ASCEND_OPP_PATH=/usr/local/Ascend/nnae/latest/opp/
#   export OPTION_EXEC_EXTERN_PLUGIN_PATH=/usr/local/Ascend/nnae/latest/fwkacllib/lib64/plugin/opskernel/libfe.so:/usr/local/Ascend/nnae/latest/fwkacllib/lib64/plugin/opskernel/libaicpu_engine.so:/usr/local/Ascend/nnae/latest/fwkacllib/lib64/plugin/opskernel/libge_local_engine.so
#   export PYTHONPATH=/usr/local/Ascend/nnae/latest/fwkacllib/python/site-packages/:/usr/local/Ascend/nnae/latest/fwkacllib/python/site-packages/auto_tune.egg/auto_tune:/usr/local/Ascend/nnae/latest/fwkacllib/python/site-packages/schedule_search.egg:$PYTHONPATH
# ############## toolkit situation ################
# else
#   export LD_LIBRARY_PATH=/usr/local/:/usr/local/lib/:/usr/lib64/:/usr/lib/:/usr/local/python3.7.5/lib/:/usr/local/openblas/lib:/home/330/ascend-toolkit/latest/fwkacllib/lib64/:/usr/local/Ascend/driver/lib64/common/:/usr/local/Ascend/driver/lib64/driver/:/usr/local/Ascend/add-ons/:/usr/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH
#   export PATH=$PATH:/home/330/ascend-toolkit/latest/fwkacllib/ccec_compiler/bin/:/home/330/ascend-toolkit/latest/toolkit/tools/ide_daemon/bin/
#   export ASCEND_OPP_PATH=/home/330/ascend-toolkit/latest/opp/
#   export OPTION_EXEC_EXTERN_PLUGIN_PATH=/home/330/ascend-toolkit/latest/fwkacllib/lib64/plugin/opskernel/libfe.so:/home/330/ascend-toolkit/latest/fwkacllib/lib64/plugin/opskernel/libaicpu_engine.so:/home/330/ascend-toolkit/latest/fwkacllib/lib64/plugin/opskernel/libge_local_engine.so
#   export PYTHONPATH=/home/330/ascend-toolkit/latest/fwkacllib/python/site-packages/:/home/330/ascend-toolkit/latest/fwkacllib/python/site-packages/auto_tune.egg/auto_tune:/home/330/ascend-toolkit/latest/fwkacllib/python/site-packages/schedule_search.egg:$PYTHONPATH
# fi

# export TASK_QUEUE_ENABLE=1
# export ASCEND_SLOG_PRINT_TO_STDOUT=0
# export ASCEND_GLOBAL_LOG_LEVEL=3
# export ASCEND_AICPU_PATH=/home/330/ascend-toolkit/latest

# path_lib=$(python3.7 -c """
# import sys
# import re
# result=''
# for index in range(len(sys.path)):
#     match_sit = re.search('-packages', sys.path[index])
#     if match_sit is not None:
#         match_lib = re.search('lib', sys.path[index])
#         if match_lib is not None:
#             end=match_lib.span()[1]
#             result += sys.path[index][0:end] + ':'
#         result+=sys.path[index] + '/torch/lib:'
# print(result)"""
# )
# export LD_LIBRARY_PATH=/usr/local/python3.7.5/lib/:${path_lib}:$LD_LIBRARY_PATH

# #设置是否开启PTCopy,0-关闭/1-开启
# export PTCOPY_ENABLE=1
# #设置是否开启combined标志,0-关闭/1-开启
# export COMBINED_ENABLE=1
# #设置特殊场景是否需要重新编译,不需要修改
# export DYNAMIC_OP="ADD#MUL"





#!/bin/bash
export install_path=/usr/local/Ascend

if [ -d ${install_path}/toolkit ]; then
    export LD_LIBRARY_PATH=/usr/include/hdf5/lib/:/usr/local/:/usr/local/lib/:/usr/lib/:${install_path}/fwkacllib/lib64/:${install_path}/driver/lib64/common/:${install_path}/driver/lib64/driver/:${install_path}/add-ons:${path_lib}:${LD_LIBRARY_PATH}
    export PATH=${install_path}/fwkacllib/ccec_compiler/bin:${install_path}/fwkacllib/bin:$PATH
    export PYTHONPATH=${install_path}/fwkacllib/python/site-packages:${install_path}/tfplugin/python/site-packages:${install_path}/toolkit/python/site-packages:$PYTHONPATH
    export PYTHONPATH=/usr/local/python3.7.5/lib/python3.7/site-packages:$PYTHONPATH
    export ASCEND_OPP_PATH=${install_path}/opp
else
    if [ -d ${install_path}/nnae/latest ];then
        export LD_LIBRARY_PATH=/usr/local/:/usr/local/python3.7.5/lib/:/usr/local/openblas/lib:/usr/local/lib/:/usr/lib64/:/usr/lib/:${install_path}/nnae/latest/fwkacllib/lib64/:${install_path}/driver/lib64/common/:${install_path}/driver/lib64/driver/:${install_path}/add-ons/:/usr/lib/aarch64_64-linux-gnu:$LD_LIBRARY_PATH
        export PATH=$PATH:${install_path}/nnae/latest/fwkacllib/ccec_compiler/bin/:${install_path}/nnae/latest/toolkit/tools/ide_daemon/bin/
        export ASCEND_OPP_PATH=${install_path}/nnae/latest/opp/
        export OPTION_EXEC_EXTERN_PLUGIN_PATH=${install_path}/nnae/latest/fwkacllib/lib64/plugin/opskernel/libfe.so:${install_path}/nnae/latest/fwkacllib/lib64/plugin/opskernel/libaicpu_engine.so:${install_path}/nnae/latest/fwkacllib/lib64/plugin/opskernel/libge_local_engine.so
        export PYTHONPATH=${install_path}/nnae/latest/fwkacllib/python/site-packages/:${install_path}/nnae/latest/fwkacllib/python/site-packages/auto_tune.egg/auto_tune:${install_path}/nnae/latest/fwkacllib/python/site-packages/schedule_search.egg:$PYTHONPATH
        export ASCEND_AICPU_PATH=${install_path}/nnae/latest
    else
        export LD_LIBRARY_PATH=/usr/local/:/usr/local/lib/:/usr/lib64/:/usr/lib/:/usr/local/python3.7.5/lib/:/usr/local/openblas/lib:${install_path}/ascend-toolkit/latest/fwkacllib/lib64/:${install_path}/driver/lib64/common/:${install_path}/driver/lib64/driver/:${install_path}/add-ons/:/usr/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH
        export PATH=$PATH:${install_path}/ascend-toolkit/latest/fwkacllib/ccec_compiler/bin/:${install_path}/ascend-toolkit/latest/toolkit/tools/ide_daemon/bin/
        export ASCEND_OPP_PATH=${install_path}/ascend-toolkit/latest/opp/
        export OPTION_EXEC_EXTERN_PLUGIN_PATH=${install_path}/ascend-toolkit/latest/fwkacllib/lib64/plugin/opskernel/libfe.so:${install_path}/ascend-toolkit/latest/fwkacllib/lib64/plugin/opskernel/libaicpu_engine.so:${install_path}/ascend-toolkit/latest/fwkacllib/lib64/plugin/opskernel/libge_local_engine.so
        export PYTHONPATH=${install_path}/ascend-toolkit/latest/fwkacllib/python/site-packages/:${install_path}/ascend-toolkit/latest/fwkacllib/python/site-packages/auto_tune.egg/auto_tune:${install_path}/ascend-toolkit/latest/fwkacllib/python/site-packages/schedule_search.egg:$PYTHONPATH
        export ASCEND_AICPU_PATH=${install_path}/ascend-toolkit/latest
    fi
fi


#将Host日志输出到串口,0-关闭/1-开启
export ASCEND_SLOG_PRINT_TO_STDOUT=0
#设置默认日志级别,0-debug/1-info/2-warning/3-error
export ASCEND_GLOBAL_LOG_LEVEL=3
#设置Event日志开启标志,0-关闭/1-开启
export ASCEND_GLOBAL_EVENT_ENABLE=0
#设置是否开启taskque,0-关闭/1-开启
export TASK_QUEUE_ENABLE=1
#设置是否开启PTCopy,0-关闭/1-开启
export PTCOPY_ENABLE=1
#设置是否开启combined标志,0-关闭/1-开启
#export COMBINED_ENABLE=1
#设置特殊场景是否需要重新编译,不需要修改
export DYNAMIC_OP="ADD#MUL"
#HCCL白名单开关,1-关闭/0-开启
export HCCL_WHITELIST_DISABLE=1
export HCCL_IF_IP=$(hostname -I |awk '{print $1}')

ulimit -SHn 512000

path_lib=$(python3.7 -c """
import sys
import re
result=''
for index in range(len(sys.path)):
    match_sit = re.search('-packages', sys.path[index])
    if match_sit is not None:
        match_lib = re.search('lib', sys.path[index])

        if match_lib is not None:
            end=match_lib.span()[1]
            result += sys.path[index][0:end] + ':'

        result+=sys.path[index] + '/torch/lib:'
print(result)"""
)

echo ${path_lib}

export LD_LIBRARY_PATH=/usr/local/python3.7.5/lib/:${path_lib}:$LD_LIBRARY_PATH





