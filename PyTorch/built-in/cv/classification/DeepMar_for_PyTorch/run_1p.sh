#!/usr/bin/env bash
source env_npu.sh

currentDir=$(cd "$(dirname "$0")";pwd)
currtime=`date +%Y%m%d%H%M%S`
train_log_dir=${currentDir}/result/training_1p_job_${currtime}
mkdir -p ${train_log_dir}
cd ${train_log_dir}
echo "train log path is ${train_log_dir}"

# Data set preprocessing
python3.7 ${currentDir}/transform_peta.py \
	--save_dir=/data/peta/ \
	--traintest_split_file=/data/peta/peta_partition.pkl

if [ $(uname -m) = "aarch64" ]
then
    taskset -c 0-47 python3.7 ${currentDir}/train_deepmar_resnet50.py \
            --save_dir=/data/peta/ \
            --workers=32 \
            --npu=0 \
            --batch_size=256 \
            --new_params_lr=0.01 \
            --finetuned_params_lr=0.01 \
            --total_epochs=150 \
            --steps_per_log=1 \
            --loss_scale 512 \
            --amp \
            --opt_level O2 > ./deepmar_1p.log 2>&1 &
else
    python3.7 ${currentDir}/train_deepmar_resnet50.py \
          --save_dir=/data/peta/ \
          --workers=32 \
          --npu=0 \
          --batch_size=256 \
          --new_params_lr=0.01 \
          --finetuned_params_lr=0.01 \
          --total_epochs=150 \
          --steps_per_log=1 \
          --loss_scale 512 \
          --amp \
          --opt_level O2 > ./deepmar_1p.log 2>&1 &
fi
