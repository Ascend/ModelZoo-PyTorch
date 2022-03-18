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
train_log_dir=${currentDir}/result/training_8p_job_${currtime}
mkdir -p ${train_log_dir}
cd ${train_log_dir}
echo "train log path is ${train_log_dir}"

KERNEL_NUM=$(($(nproc)/8))
for i in $(seq 0 7)
do
    if [ $(uname -m) = "aarch64" ]
    then
        PID_START=$((KERNEL_NUM * i))
        PID_END=$((PID_START + KERNEL_NUM - 1))
        taskset -c $PID_START-$PID_END python3.7 ${currentDir}/main_8p.py \
            --cfg ${currentDir}/LMDB_8p_config.yaml \
            --npu ${i} > ./crnn_8p_${i}_npu.log 2>&1 &
    else
        python3.7 ${currentDir}/main_8p.py \
            --cfg ${currentDir}/LMDB_8p_config.yaml \
            --npu ${i} > ./crnn_8p_${i}_npu.log 2>&1 &
    fi
done
