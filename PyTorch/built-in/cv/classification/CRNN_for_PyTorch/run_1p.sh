#!/usr/bin/env bash
source env_npu.sh
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_LOG_LEVEL=3
export TASK_QUEUE_ENABLE=1
export PTCOPY_ENABLE=1
export COMBINED_ENABLE=1
export SWITCH_MM_OUTPUT_ENABLE=1


currentDir=$(cd "$(dirname "$0")";pwd)
currtime=`date +%Y%m%d%H%M%S`
train_log_dir=${currentDir}/result/training_1p_job_${currtime}
mkdir -p ${train_log_dir}
cd ${train_log_dir}
echo "train log path is ${train_log_dir}"

python3.7 ${currentDir}/main.py --cfg ${currentDir}/LMDB_config.yaml > ./crnn_1p.log 2>&1 &