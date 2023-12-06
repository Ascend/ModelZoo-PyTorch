#!bin/bash
# 网络名称，同目录名称
Network="MAPPO_for_PyTorch"
WORLD_SIZE=1
BATCH_SIZE=1

ASCEND_DEVICE_ID=0

# 校验是否指定了device_id,分动态分配device_id与手动指定device_id,此处不需要修改
if [ $ASCEND_DEVICE_ID ];then
    echo "device id is ${ASCEND_DEVICE_ID}"
else
    "[Error] device id must be config"
    exit 1
fi

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

echo ${test_path_dir}
source ${test_path_dir}/env_npu.sh

#创建DeviceID输出目录，不需要修改
output_path=${cur_path}/test/output/${ASCEND_DEVICE_ID}

if [ -d ${output_path} ]; then
  rm -rf ${output_path}
fi

mkdir -p ${output_path}

cd onpolicy/scripts/train_mpe_scripts
chmod +x ./train_mpe_comm.sh
#为确保性能，只允许同时运行一个mappo脚本，如有需要运行多任务请注释pkill代码
pkill -9 mappo
wait
#训练开始时间，不需要修改
start_time=$(date +%s)
echo "start_time: ${start_time}"

RANK_ID=0
KERNEL_NUM=$(($(nproc)/8))
PID_START=$((KERNEL_NUM * RANK_ID))
PID_END=$((PID_START + KERNEL_NUM - 1))

exps=$(seq 1 10)
for exp in $exps
do
  #为确保性能，只允许同时运行一个mappo脚本，如有需要运行多任务请注释pkill代码
  pkill -9 mappo
  wait
  taskset -c $PID_START-$PID_END sh train_mpe_comm.sh >> ${output_path}/train_${ASCEND_DEVICE_ID}.log 2>&1 &
  wait
done


# 训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(($end_time - $start_time))

# 训练用例信息，不需要修改
BatchSize=${BATCH_SIZE}
DeviceType=$(uname -m)
CaseName=${Network}_bs${BatchSize}_${WORLD_SIZE}'p'_'acc'

# 结果打印，不需要修改
echo "------------------ Final result ------------------"
# 输出性能FPS，需要模型审视修改
grep "total num timesteps " ${output_path}/train_${ASCEND_DEVICE_ID}.log | awk -F 'current FPS ' '{print $2}' | awk -F '.' '{print $1}' >${output_path}/train_${ASCEND_DEVICE_ID}_fps.log
FPS=$(cat ${output_path}/train_${ASCEND_DEVICE_ID}_fps.log | sort -n | tail -5000 | awk '{a+=$1} END {if (NR != 0) printf("%.2f", a/NR)}')
# 打印，不需要修改
echo "Final Performance images/sec : $FPS"

# 输出训练精度,需要模型审视修改
grep "Eval average episode rewards of agent: " ${output_path}/train_${ASCEND_DEVICE_ID}.log | awk -F "agent: " '{print $2}' &>${output_path}/train_${ASCEND_DEVICE_ID}_acc.log
train_accuracy=$(cat ${output_path}/train_${ASCEND_DEVICE_ID}_acc.log | sort -n | tail -8 | awk '{a+=$1} END {if (NR != 0) printf("%.3f", a/NR)}')
train_accuracy=$(echo $train_accuracy | awk '{printf ("%.1f\n",$1)}')
# 打印，不需要修改
echo "Final Train Accuracy : ${train_accuracy}"
echo "E2E Training Duration sec : $e2e_time"

# 性能看护结果汇总
# 获取性能数据，不需要修改
# 吞吐量
ActualFPS=${FPS}
# 单迭代训练时长
TrainingTime=$(awk 'BEGIN{printf "%.2f\n", '${BATCH_SIZE}'*25*128/'${FPS}'}')

# 关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" >${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${WORLD_SIZE}" >>${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >>${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >>${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >>${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >>${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >>${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainAccuracy = ${train_accuracy}" >>${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >>${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log

