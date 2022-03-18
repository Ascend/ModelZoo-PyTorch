#!/usr/bin/env bash
source env_npu.sh
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_LOG_LEVEL=3
export TASK_QUEUE_ENABLE=1
export PTCOPY_ENABLE=1


currentDir=$(cd "$(dirname "$0")";pwd)
currtime=`date +%Y%m%d%H%M%S`
train_log_dir=${currentDir}/result/training_8p_job_${currtime}
mkdir -p ${train_log_dir}
cd ${train_log_dir}
echo "train log path is ${train_log_dir}"

for i in $(seq 0 3)
do
python3.7 ${currentDir}/main_anycard.py --cfg ${currentDir}/LMDB_anycard_config.yaml --npu ${i} > ./crnn_8p_${i}_npu.log 2>&1 &
done
