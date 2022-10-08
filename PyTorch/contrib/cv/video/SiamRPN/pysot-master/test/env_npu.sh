#!/bin/bash
CANN_INSTALL_PATH_CONF='/etc/Ascend/ascend_cann_install.info'

if [ -f $CANN_INSTALL_PATH_CONF ]; then
    CANN_INSTALL_PATH=$(cat $CANN_INSTALL_PATH_CONF | grep Install_Path | cut -d "=" -f 2)
else
    CANN_INSTALL_PATH="/usr/local/Ascend"
fi

if [ -d ${CANN_INSTALL_PATH}/ascend-toolkit/latest ]; then
    source ${CANN_INSTALL_PATH}/ascend-toolkit/set_env.sh
else
    source ${CANN_INSTALL_PATH}/nnae/set_env.sh
fi
msnpureport -g error -d 0
msnpureport -g error -d 1
msnpureport -g error -d 2
msnpureport -g error -d 3
msnpureport -g error -d 4
msnpureport -g error -d 5
msnpureport -g error -d 6
msnpureport -g error -d 7

#��Host��־���������,0-�ر�/1-����
export ASCEND_SLOG_PRINT_TO_STDOUT=1
#����Ĭ����־����,0-debug/1-info/2-warning/3-error
export ASCEND_GLOBAL_LOG_LEVEL=3
#����Event��־������־,0-�ر�/1-����
export ASCEND_GLOBAL_EVENT_ENABLE=0
#�����Ƿ���taskque,0-�ر�/1-����
export TASK_QUEUE_ENABLE=0
#�����Ƿ���PTCopy,0-�ر�/1-����
export PTCOPY_ENABLE=1
#�����Ƿ���combined��־,0-�ر�/1-����
export COMBINED_ENABLE=0
#�������ⳡ���Ƿ���Ҫ���±���,����Ҫ�޸�
export DYNAMIC_OP="ADD#MUL"
#HCCL����������,1-�ر�/0-����
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
