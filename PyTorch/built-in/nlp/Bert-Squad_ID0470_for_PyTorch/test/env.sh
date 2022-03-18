#!/bin/bash
cur_path=`pwd`/../
export install_path=/usr/local/Ascend 
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/common/:/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH # 仅容器训练场景配置
export PATH=${install_path}/fwkacllib/ccec_compiler/bin:${install_path}/fwkacllib/bin:$PATH
export LD_LIBRARY_PATH=${install_path}/fwkacllib/lib64:$LD_LIBRARY_PATH
export PYTHONPATH=${install_path}/fwkacllib/python/site-packages:$PYTHONPATH
export PYTHONPATH=/usr/local/python3.7.5/lib/python3.7/site-packages:${install_path}/tfplugin/python/site-packages:$PYTHONPATH
export ASCEND_OPP_PATH=${install_path}/opp
export ASCEND_AICPU_PATH=${install_path}
export PYTHONPATH=$cur_path/models/research:$cur_path/models/research/slim:$PYTHONPATH
export JOB_ID=10087
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_DEVICE_ID=0
export BMMV2_ENABLE=1