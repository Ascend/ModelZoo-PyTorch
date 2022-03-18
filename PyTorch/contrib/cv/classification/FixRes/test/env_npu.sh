export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/fwkacllib/lib64:/usr/local/Ascend/ascend-toolkit/latest/atc/lib64:$LD_LIBRARY_PATH
export PYTHONPATH=/usr/local/Ascend/ascend-toolkit/latest/fwkacllib/python/site-packages:/usr/local/Ascend/ascend-toolkit/latest/toolkit/python/site-packages:/usr/local/Ascend/ascend-toolkit/latest/atc/python/site-packages:/usr/local/Ascend/ascend-toolkit/latest/pyACL/python/site-packages/acl:$PYTHONPATH
export PATH=/usr/local/Ascend/ascend-toolkit/latest/fwkacllib/ccec_compiler/bin:/usr/local/Ascend/ascend-toolkit/latest/fwkacllib/bin:/usr/local/Ascend/ascend-toolkit/latest/atc/bin:/usr/local/Ascend/ascend-toolkit/latest/atc/ccec_compiler/bin:$PATH
export ASCEND_AICPU_PATH=/usr/local/Ascend/ascend-toolkit/latest
export ASCEND_OPP_PATH=/usr/local/Ascend/ascend-toolkit/latest/opp
export TOOLCHAIN_HOME=/usr/local/Ascend/ascend-toolkit/latest/toolkit

export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_GLOBAL_EVENT_ENABLE=0
export TASK_QUEUE_ENABLE=1
export PTCOPY_ENABLE=1
# export COMBINED_ENABLE=0
export COMBINED_ENABLE=1
export DYNAMIC_OP='ADD#MUL'
export HCCL_WHITELIST_DISABLE=1