#!/bin/bash
CANN_INSTALL_PATH_CONF='/etc/Ascend/ascend_cann_install.info'

if [ -f $CANN_INSTALL_PATH_CONF ]; then
    DEFAULT_CANN_INSTALL_PATH=$(cat $CANN_INSTALL_PATH_CONF | grep Install_Path | cut -d "=" -f 2)
else
    DEFAULT_CANN_INSTALL_PATH="/usr/local/Ascend"
fi

if [ -d ${DEFAULT_CANN_INSTALL_PATH}/ascend-toolkit/latest ]; then
    source ${DEFAULT_CANN_INSTALL_PATH}/ascend-toolkit/set_env.sh
else
    source ${DEFAULT_CANN_INSTALL_PATH}/nnae/set_env.sh
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
export COMBINED_ENABLE=1
#设置特殊场景是否需要重新编译,不需要修改
export DYNAMIC_OP="ADD#MUL"
#HCCL白名单开关,1-关闭/0-开启
export HCCL_WHITELIST_DISABLE=1
export HCCL_IF_IP=$(hostname -I |awk '{print $1}')

#设置device侧日志登记为error
${install_path}/driver/tools/msnpureport -g error -d 0
${install_path}/driver/tools/msnpureport -g error -d 1
${install_path}/driver/tools/msnpureport -g error -d 2
${install_path}/driver/tools/msnpureport -g error -d 3
${install_path}/driver/tools/msnpureport -g error -d 4
${install_path}/driver/tools/msnpureport -g error -d 5
${install_path}/driver/tools/msnpureport -g error -d 6
${install_path}/driver/tools/msnpureport -g error -d 7
#关闭Device侧Event日志
${install_path}/driver/tools/msnpureport -e disable

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
