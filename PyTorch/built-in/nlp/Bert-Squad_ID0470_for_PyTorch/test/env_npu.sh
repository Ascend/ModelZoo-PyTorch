/usr/local/Ascend/driver/tools/msnpureport -g error -d 0
/usr/local/Ascend/driver/tools/msnpureport -g error -d 4

export LD_LIBRARY_PATH=/usr/include/hdf5/lib/:$LD_LIBRARY_PATH
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


export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_GLOBAL_EVENT_ENABLE=0
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export TASK_QUEUE_ENABLE=1
export BMMV2_ENABLE=1
export COMBINED_ENABLE=1
export PTCOPY_ENABLE=1
export HCCL_WHITELIST_DISABLE=1
export SCALAR_TO_HOST_MEM=1

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