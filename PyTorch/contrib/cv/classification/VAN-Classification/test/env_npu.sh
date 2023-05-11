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
