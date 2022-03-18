#!/usr/bin/env bash
source env_npu.sh
ip=$(hostname -I|awk '{print $1}')
export ASCEND_SLOG_PRINT_TO_STDOUT=0
su HwHiAiUser -c "adc --host ${ip}:22118 --log \"SetLogLevel(0)[error]\" --device 7"
currentDir=$(cd "$(dirname "$0")";pwd)
currtime=`date +%Y%m%d%H%M%S`
eval_log_dir=${currentDir}/result/eval_1p_job_${currtime}
mkdir -p ${eval_log_dir}
cd ${eval_log_dir}
echo "eval log path is ${eval_log_dir}"

export TASK_QUEUE_ENABLE=1
python3.7 ${currentDir}/densenet121_1p_main.py \
	--workers 40 \
	--arch densenet121 \
	--npu 7 \
	--lr 0.1 \
	--momentum 0.9 \
	--amp \
	--batch-size 256 \
	--epoch 90 \
	--evaluate \
	--resume checkpoint.pth.tar \
	--data /data/imagenet/ > ./densenet121_eval.log 2>&1 &