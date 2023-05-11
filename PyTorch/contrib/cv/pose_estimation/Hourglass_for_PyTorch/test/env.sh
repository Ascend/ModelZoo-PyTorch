#!/bin/bash
# 配置CANN相关环境变量
CANN_INSTALL_PATH_CONF='/etc/Ascend/ascend_cann_install.info'

if [ -f $CANN_INSTALL_PATH_CONF ]; then
  CANN_INSTALL_PATH=$(cat $CANN_INSTALL_PATH_CONF | grep Install_Path | cut -d "=" -f 2)
else
  CANN_INSTALL_PATH="/usr/local/Ascend/"
fi

if [ -d ${CANN_INSTALL_PATH}/ascend-toolkit/latest ];then
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

