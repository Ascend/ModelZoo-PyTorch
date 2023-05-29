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

#����device����־�Ǽ�Ϊerror
msnpureport -g error -d 0
msnpureport -g error -d 1
msnpureport -g error -d 2
msnpureport -g error -d 3
msnpureport -g error -d 4
msnpureport -g error -d 5
msnpureport -g error -d 6
msnpureport -g error -d 7
#�ر�Device��Event��־
msnpureport -e disable


#��Host��־���������,0-�ر�/1-����
export ASCEND_SLOG_PRINT_TO_STDOUT=0
#����Ĭ����־����,0-debug/1-info/2-warning/3-error
export ASCEND_GLOBAL_LOG_LEVEL==3
#����Event��־������־,0-�ر�/1-����
export ASCEND_GLOBAL_EVENT_ENABLE=0
#�����Ƿ���taskque,0-�ر�/1-����
export TASK_QUEUE_ENABLE=1
#�����Ƿ���PTCopy,0-�ر�/1-����
export PTCOPY_ENABLE=1
#�����Ƿ���2��������combined��־,0-�ر�/1-����
export COMBINED_ENABLE=1
#�������ⳡ���Ƿ���Ҫ���±���,����Ҫ�޸�
export DYNAMIC_OP="ADD#MUL"
# HCCL����������,1-�ر�/0-����
export HCCL_WHITELIST_DISABLE=1
# HCCLĬ�ϳ�ʱʱ��120s���٣��޸�Ϊ1800s����PyTorchĬ������
export HCCL_CONNECT_TIMEOUT=1800

ulimit -SHn 512000

