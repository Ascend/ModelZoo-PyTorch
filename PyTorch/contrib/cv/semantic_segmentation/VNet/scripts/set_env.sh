export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/Ascend/ascend-toolkit/latest/fwkacllib/lib64:/usr/local/Ascend/ascend-toolkit/latest/atc/lib64
export PYTHONPATH=$PYTHONPATH:/usr/local/Ascend/ascend-toolkit/latest/fwkacllib/python/site-packages:/usr/local/Ascend/ascend-toolkit/latest/toolkit/python/site-packages:/usr/local/Ascend/ascend-toolkit/latest/atc/python/site-packages:/usr/local/Ascend/ascend-toolkit/latest/pyACL/python/site-packages/acl
export PATH=$PATH:/usr/local/Ascend/ascend-toolkit/latest/fwkacllib/ccec_compiler/bin:/usr/local/Ascend/ascend-toolkit/latest/fwkacllib/bin:/usr/local/Ascend/ascend-toolkit/latest/atc/bin
export ASCEND_AICPU_PATH=/usr/local/Ascend/ascend-toolkit/latest
export ASCEND_OPP_PATH=/usr/local/Ascend/ascend-toolkit/latest/opp
export TOOLCHAIN_HOME=/usr/local/Ascend/ascend-toolkit/latest/toolkit

export HCCL_WHITELIST_DISABLE=1
export TASK_QUEUE_ENABLE=1
#设置是否开启PTCopy,0-关闭/1-开启
export PTCOPY_ENABLE=1
#设置是否开启combined标志,0-关闭/1-开启
export COMBINED_ENABLE=1