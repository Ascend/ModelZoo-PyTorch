# main env
currentDir=$(cd "$(dirname "$0")"; pwd)
path_lib=$(python3.7 -c """
import sys
import re
result=''
for index in range(len(sys.path)):
    match_sit = re.search('site', sys.path[index])
    if match_sit is not None:
        match_lib = re.search('lib', sys.path[index])

        if match_lib is not None:
            end=match_lib.span()[1]
            result += sys.path[index][0:end] + ':'

        result+=sys.path[index] + '/torch/lib:'
print(result)"""
)

echo ${path_lib}
export DYNAMIC_OP="ADD#MUL"
export install_path=/usr/local/Ascend
export LD_LIBRARY_PATH=/usr/local/:/usr/local/lib/:/usr/lib/:${install_path}/fwkacllib/lib64/:${install_path}/driver/lib64/common/:${install_path}/driver/lib64/driver/:${install_path}/add-ons:${path_lib}:${LD_LIBRARY_PATH}
export PYTHONPATH=$PYTHONPATH:${install_path}/opp/op_impl/built-in/ai_core/tbe:${install_path}/fwkacllib/python/site-packages/hccl:${install_path}/fwkacllib/python/site-packages/te:${install_path}/fwkacllib/python/site-packages/topi:${install_path}/tfplugin/python/site-packages
export PATH=${install_path}/fwkacllib/ccec_compiler/bin:${install_path}/fwkacllib/bin:${PATH}
export ASCEND_OPP_PATH=${install_path}/opp

export SOC_VERSION=Ascend910
export HCCL_CONNECT_TIMEOUT=600

export SLOG_PRINT_TO_STDOUT=1
export TASK_QUEUE_ENABLE=1
export PTCOPY_ENABLE=1
export ASCEND_GLOBAL_LOG_LEVEL=1
export ASCEND_SLOG_PRINT_TO_STDOUT=1

# user env
export JOB_ID=crnn_pytorch_8p
export RANK_TABLE_FILE=/autotest/CI_daily/PyTorch_CRNN/result/cloud-localhost-20210224162017-0/config/hccl.json
export RANK_SIZE=1
export RANK_INDEX=2
export RANK_ID=cloud-localhost-20210224162017-0

# profiling env
export PROFILING_MODE=false
export PROFILING_OPTIONS=\{\"result_path\":\"/var/log/npu/profiling\"\,\"training_trace\":\"on\"\,\"task_trace\":\"on\"\,\"aicpu_trace\":\"on\"\,\"fp_point\":\"\"\,\"bp_point\":\"\"\,\"ai_core_metrics\":\"PipeUtilization\"\}
#export AICPU_PROFILING_MODE={AICPU_PROFILING_MODE}
#export PROFILING_OPTIONS={PROFILING_OPTIONS}
#export FP_POINT=
#export BP_POINT=

#export DUMP_GE_GRAPH=1
#export DUMP_GRAPH_LEVEL=3

# debug env
#export DUMP_GE_GRAPH=2
#export DUMP_OP=1
#export DUMP_OP_LESS=1
#export PRINT_MODEL=1
#export TE_PARALLEL_COMPILER=0

# system env
#ulimit -c unlimited
