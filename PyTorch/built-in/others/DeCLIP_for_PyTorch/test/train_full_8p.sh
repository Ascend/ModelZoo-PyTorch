#!/bin/bash

#网络名称，同目录名称
Network="DeCLIP_for_PyTorch"
batch_size=128

###############指定训练脚本执行路径###############
# cd到与test文件夹同层级目录下执行脚本，提高兼容性；test_path_dir为包含test文件夹的路径
cur_path=$(pwd)
cur_path_last_dirname=${cur_path##*/}
if [ x"${cur_path_last_dirname}" == x"test" ]; then
	test_path_dir=${cur_path}
	cd ..
	cur_path=$(pwd)
else
	test_path_dir=${cur_path}/test
fi

ASCEND_DEVICE_ID=0

#################创建日志输出目录，不需要修改#################
if [ -d ${test_path_dir}/output/${ASCEND_DEVICE_ID} ]; then
	rm -rf ${test_path_dir}/output/${ASCEND_DEVICE_ID}
	mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID
else
	mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID
fi

#训练开始时间，不需要修改
start_time=$(date +%s)
# 非平台场景时source 环境变量
check_etp_flag=$(env | grep etp_running_flag)
etp_flag=$(echo ${check_etp_flag#*=})
if [ x"${etp_flag}" != x"true" ]; then
	source ${test_path_dir}/env_npu.sh
fi

export PYTHONPATH=./DeCLIP:$PYTHONPATH
export SLURM_NTASKS=8
export SLURM_NODELIST="        127.0.0.1"
export MASTER_PORT="23333"

RANK_ID_START=0
RANK_SIZE=$SLURM_NTASKS

KERNEL_NUM=$(($(nproc) / 8))
for ((RANK_ID = $RANK_ID_START; RANK_ID < $((RANK_SIZE + RANK_ID_START)); RANK_ID++)); do
	PID_START=$((KERNEL_NUM * $RANK_ID))
	PID_END=$((PID_START + KERNEL_NUM - 1))

	export SLURM_PROCID=$RANK_ID
	nohup taskset -c $PID_START-$PID_END python3 -u runner.py --config config.yaml >${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &

done

wait

#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(($end_time - $start_time))

#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
FPS=$(grep "CRITICAL" ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | tail -n 50 | awk -F 'FPS ' '{print $2}' | awk '{print $1}' | awk '{sum+=$1} END {print sum/NR}' )

#打印，不需要修改
echo "Final Performance images/sec : $FPS"

#输出训练精度,需要模型审视修改
train_accuracy=$(grep "top1" ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | awk 'END {print $2}'| awk -F "," '{print $1}')

#打印，不需要修改
echo "Final Train Accuracy : ${train_accuracy}"
echo "E2E Training Duration sec : $e2e_time"

#性能看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=$(uname -m)
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'acc'

##获取性能数据，不需要修改
#吞吐量
ActualFPS=${FPS}
#单迭代训练时长
TrainingTime=$(awk 'BEGIN{printf "%.2f\n", '${batch_size}'*1000/'${FPS}'}')

#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
grep "Loss" ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | awk 'END {print}' | awk -F "(" '{print $5}' | awk -F ")" '{print $1}' >${test_path_dir}/output/${ASCEND_DEVICE_ID}//train_${CaseName}_loss.txt
#最后一个迭代loss值，不需要修改
ActualLoss=$(awk 'END {print}' ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${CaseName}_loss.txt)

#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" >${test_path_dir}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >>${test_path_dir}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "BatchSize = ${BatchSize}" >>${test_path_dir}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "DeviceType = ${DeviceType}" >>${test_path_dir}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "CaseName = ${CaseName}" >>${test_path_dir}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >>${test_path_dir}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >>${test_path_dir}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >>${test_path_dir}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >>${test_path_dir}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "TrainAccuracy = ${train_accuracy}" >>${test_path_dir}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
