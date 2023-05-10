#!/bin/bash
source ./test/env_npu.sh

#当前路径,不需要修改
cur_path=$(pwd)
cur_path_last_dirname=${cur_path##*/}
if [ x"${cur_path_last_dirname}" == x"test" ]; then
  test_path_dir=${cur_path}
  cd ..
  cur_path=$(pwd)
else
  test_path_dir=${cur_path}/test
fi

#网络名称,同目录名称,需要模型审视修改
Network="DeepLabV3_for_PyTorch"

# 指定训练所使用的npu device卡id
device_id=0
# 校验是否指定了device_id,分动态分配device_id与手动指定device_id,此处不需要修改
if [ ${device_id} ]; then
  export ASCEND_DEVICE_ID=${device_id}
  echo "device id is ${ASCEND_DEVICE_ID}"
else
  "[Error] device id must be config"
  exit 1
fi

#创建DeviceID输出目录，不需要修改
if [ -d $test_path_dir/output/${ASCEND_DEVICE_ID} ]; then
  rm -rf $test_path_dir/output/$ASCEND_DEVICE_ID
  mkdir -p $test_path_dir/output/$ASCEND_DEVICE_ID
else
  mkdir -p $test_path_dir/output/$ASCEND_DEVICE_ID
fi

#################启动训练脚本#################
start_time=$(date +%s)
nohup bash tools/dist_train.sh \
  configs/deeplabv3/deeplabv3_r101-d16-mg124_8xb8-40k_cityscapes-512x1024.py \
  8 --cfg-options train_cfg.max_iters=2000 \
  train_cfg.val_interval=1000 >${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &
wait

##################获取训练数据################
#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(($end_time - $start_time))

#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
FPS=$(grep "FPS:" ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | grep "train" | awk -F "FPS:" '{print $2}' | awk 'END {print $1}')

#打印，不需要修改
echo "Final Performance images/sec : $FPS"

#输出训练精度,需要模型审视修改
train_accuracy=$(grep "mIoU:" ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | awk -F "mIoU:" '{print $2}' | awk 'END {print $1}')

#打印，不需要修改
echo "Final Train Accuracy : ${train_accuracy}"
echo "E2E Training Duration sec : $e2e_time"

#性能看护结果汇总
#训练用例信息，不需要修改
BatchSize=64 #仅用于打印，不在训练中生效
WORLD_SIZE=1
DeviceType=$(uname -m)
CaseName=${Network}_bs${BatchSize}_'p'_'perf'

##获取性能数据，不需要修改
#吞吐量
ActualFPS=${FPS}
#单迭代训练时长
TrainingTime=$(awk 'BEGIN{printf "%.2f\n", '${BatchSize}'*1000/'${FPS}'}')

#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
grep "loss:" ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | awk -F "loss:" '{print $2}'| awk '{print $1}' >${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${CaseName}_loss.txt
#最后一个迭代loss值，不需要修改
ActualLoss=$(awk 'END {print}' ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${CaseName}_loss.txt)

#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" >${test_path_dir}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "RankSize = ${WORLD_SIZE}" >>${test_path_dir}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "BatchSize = ${BatchSize}" >>${test_path_dir}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "DeviceType = ${DeviceType}" >>${test_path_dir}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "CaseName = ${CaseName}" >>${test_path_dir}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >>${test_path_dir}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >>${test_path_dir}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >>${test_path_dir}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >>${test_path_dir}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "TrainAccuracy = ${train_accuracy}" >>${test_path_dir}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
