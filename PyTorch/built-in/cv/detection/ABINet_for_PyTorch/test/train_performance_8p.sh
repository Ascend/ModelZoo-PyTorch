#!/bin/bash
################基础配置参数，需要模型审视修改##################
# 网络名称，同目录名称
Network="ABINet_for_PyTorch"
BATCH_SIZE=192
WORLD_SIZE=8
WORK_DIR="abinet"
LOAD_FROM=""
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

for para in $*; do
  if [[ $para == --work_dir* ]]; then
    WORK_DIR=$(echo ${para#*=})
  elif [[ $para == --load_from* ]]; then
    LOAD_FROM=$(echo ${para#*=})
  fi
done

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

source ${test_path_dir}/env_npu.sh

ASCEND_DEVICE_ID=0

if [ -d ${cur_path}/test/output/${ASCEND_DEVICE_ID} ]; then
  rm -rf ${cur_path}/test/output/${ASCEND_DEVICE_ID}
  mkdir -p ${cur_path}/test/output/${ASCEND_DEVICE_ID}
else
  mkdir -p ${cur_path}/test/output/${ASCEND_DEVICE_ID}
fi

start_time=$(date +%s)
# 非平台场景时source 环境变量
check_etp_flag=$(env | grep etp_running_flag)
etp_flag=$(echo ${check_etp_flag#*=})
if [ x"${etp_flag}" != x"true" ]; then
  source ${test_path_dir}/env_npu.sh
fi

python -m torch.distributed.launch \
  --nnodes=$NNODES \
  --master_addr=$MASTER_ADDR \
  --nproc_per_node=8 \
  --master_port=$PORT \
  tools/train.py \
  configs/textrecog/abinet/abinet_academic_1000iters.py \
  --seed 0 \
  --load-from=${LOAD_FROM} \
  --work-dir=${WORK_DIR} \
  --launcher pytorch \
  >$cur_path/test/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &
wait

##################获取训练数据################
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
grep "time:" ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | awk -F "time:" '{print substr($2,0,6)}' &>${test_path_dir}/output/$ASCEND_DEVICE_ID/train_${CaseName}_fps.log
FPS=$(cat ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_${CaseName}_fps.log | sort -n | head -100 | awk -v bs=$BATCH_SIZE -v ws=$WORLD_SIZE '{a+=$1} END {if (NR != 0) printf("%.3f", 1/a*NR*bs*ws)}')
# 打印，不需要修改
echo "Final Performance images/sec : $FPS"

# 打印，不需要修改
echo "E2E Training Duration sec : $e2e_time"

# 性能看护结果汇总
# 获取性能数据，不需要修改
# 吞吐量
ActualFPS=${FPS}
# 单迭代训练时长
TrainingTime=$(awk 'BEGIN{printf "%.2f\n", '${BATCH_SIZE}'*1000/'${FPS}'}')

# 从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
grep "Iter " ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log | grep -v Test | awk -F "loss:" '{print $NF}' | awk -F " " '{print $1}' >>${test_path_dir}/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt
# 最后一个迭代loss值，不需要修改
ActualLoss=$(awk 'END {print}' ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt)

# 关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" >${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${WORLD_SIZE}" >>${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >>${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >>${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >>${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >>${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >>${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >>${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
