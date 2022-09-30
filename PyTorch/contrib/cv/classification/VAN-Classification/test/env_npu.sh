#!/use/bin/bash

#ASCEND_HOME=/home/wxk/pack/runp/ascend-toolkit/latest
ASCEND_HOME=/usr/local/Ascend/ascend-toolkit/latest
export LD_LIBRARY_PATH=/usr/local/:/usr/local/lib/:/usr/lib64/:/usr/lib/:/usr/local/python3.7.5/lib/:/usr/local/openblas/lib:$ASCEND_HOME/fwkacllib/lib64/:/usr/local/Ascend/driver/lib64/common/:/usr/local/Ascend/driver/lib64/driver/:/usr/local/Ascend/add-ons/:/usr/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH
export PATH=$PATH:$ASCEND_HOME/fwkacllib/ccec_compiler/bin/:$ASCEND_HOME/toolkit/tools/ide_daemon/bin/
export ASCEND_OPP_PATH=$ASCEND_HOME/opp/
export OPTION_EXEC_EXTERN_PLUGIN_PATH=$ASCEND_HOME/fwkacllib/lib64/plugin/opskernel/libfe.so:$ASCEND_HOME/fwkacllib/lib64/plugin/opskernel/libaicpu_engine.so:$ASCEND_HOME/fwkacllib/lib64/plugin/opskernel/libge_local_engine.so
export PYTHONPATH=$ASCEND_HOME/fwkacllib/python/site-packages/:$ASCEND_HOME/fwkacllib/python/site-packages/auto_tune.egg/auto_tune:$ASCEND_HOME/fwkacllib/python/site-packages/schedule_search.egg:$PYTHONPATH

export TASK_QUEUE_ENABLE=1
#设置是否开启PTCopy,0-关闭/1-开启
export PTCOPY_ENABLE=1
#设置是否开启2个非连续combined标志,0-关闭/1-开启
export COMBINED_ENABLE=1
#设置是否开启3个非连续combined标志,0-关闭/1-开启
export TRI_COMBINED_ENABLE=1
#设置特殊场景是否需要重新编译,不需要修改
export DYNAMIC_OP="ADD#MUL"
#默认hccl建链时间120
#export HCCL_CONNECT_TIMEOUT=1800
# 对算子add和mul优化，不加可能导致模型性能降低
export SCALAR_TO_HOST_MEM=1
# HCCL白名单开关,1-关闭/0-开启
export HCCL_WHITELIST_DISABLE=1
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_LOG_LEVEL=3
export MM_BMM_ND_ENABLE=1
export bmmv2_enable=1
export BMMV2_ENABLE=1

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
export LD_LIBRARY_PATH=/usr/local/python3.7.5/lib/:${path_lib}:$LD_LIBRARY_PATH
