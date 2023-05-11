#!/bin/bash

#集合通信参数,不需要修改
export RANK_SIZE=8
export MASTER_ADDR='127.0.0.1'
export MASTER_PORT='18888'

#网络名称,同目录名称,需要模型审视修改
Network="BiseNetV1_for_Pytorch"

#训练batch_size,需要模型审视修改
batch_size=8

#训练step数,不需要修改
train_steps=40000

#维持参数，以下不需要修改
device_id=0

##################指定训练脚本执行路径##################
# cd到与test文件同层级目录下执行脚本，提高兼容性；test_path_dir为包含test文件夹的路径
cur_path=`pwd`
cur_path_last_dirname=${cur_path##*/}
if [ x"${cur_path_last_dirname}" == x"test" ]; then
    test_path_dir=${cur_path}
    cd ..
    cur_path=`pwd`
else
    test_path_dir=${cur_path}/test
fi

#检查divice_id设置，不需要修改
if [ $ASCEND_DEVICE_ID ];then
    echo "device id is ${ASCEND_DEVICE_ID}"
elif [ ${device_id} ];then
    export ASCEND_DEVICE_ID=${device_id}
    echo "device id is ${ASCEND_DEVICE_ID}"
else
    "[ERROR] device id must be config"
    exit 1
fi

#创建DeviceID输出目录，不需要修改
if [ -d ${test_path_dir}/output/${ASCEND_DEVICE_ID} ];then
    rm -rf ${test_path_dir}/output/${ASCEND_DEVICE_ID}
    mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID/
else
    mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID/
fi

# 非平台场景时source 环境变量
check_etp_flag=`env | grep etp_running_flag`
etp_flag=`echo ${check_etp_flag#*=}`
if [ x"${etp_flag}" != x"true" ];then
    source ${test_path_dir}/env_npu.sh
else
    xcit_main_dirname=$(basename ${more_path1})
    if [ -f /root/.cache/torch/hub/${xcit_main_dirname} ]; then
        echo "${xcit_main_dirname} file exists"
    else
        mkdir -p /root/.cache/torch/hub/
        cp -r ${more_path1} /root/.cache/torch/hub/
    fi
fi

#设置环境变量，不需要修改
RANK_ID=$ASCEND_DEVICE_ID
export RANK_ID=$RANK_ID

#训练开始时间，不需要修改
start_time=$(date +%s)

# 创建work_dirs文件夹，不需要修改
mkdir -p work_dirs

#执行训练脚本，以下传参不需要修改，其他需要模型审视修改
export WORLD_SIZE=${RANK_SIZE}
KERNEL_NUM=$(($(nproc)/8))

for((RANK_ID=0;RANK_ID<RANK_SIZE;RANK_ID++))
do
  export OMP_NUM_THREADS=1
  export RANK=$RANK_ID
  export LOCAL_RANK=$RANK_ID
  PID_START=$((KERNEL_NUM * RANK_ID))
  PID_END=$((PID_START + KERNEL_NUM - 1))
  nohup taskset -c $PID_START-$PID_END python3 -u tools/train.py configs/bisenetv1/bisenetv1_r50-d32_4x4_1024x1024_160k_cityscapes_8p.py \
    --launcher pytorch \
    --work-dir=work_dirs/bisenetv1_8p_train \
    --seed 36436756 \
    --opt-level O1 > ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}_8p.log 2>&1 &
done
wait

#8p情况下仅0卡(主节点)有完整日志,因此后续日志提取仅涉及0卡
ASCEND_DEVICE_ID=0

#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
time=`grep "mmseg - INFO - Iter " ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}_8p.log | tail -n 205 | head -n 200 | awk -F "time:" '{print $2}' | awk -F "," '{print $1}' | awk '{sum+=$1} END {print sum/200}'`
FPS=`awk 'BEGIN{printf "%.2f\n", '${batch_size}'/'${time}'*8}'`
#打印，不需要修改
echo "Final Performance images/sec : $FPS"

#输出训练精度,需要模型审视修改
train_accuracy=`grep "mmseg - INFO - Iter(val)" ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}_8p.log | tail -n 1 | awk -F "mIoU: " '{print $2}' | awk -F "," '{print $1 * 100}'`
#打印，不需要修改
echo "Final Train Accuracy : ${train_accuracy}"
echo "E2E Training Duration sec : $e2e_time"

#稳定性精度看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'acc'

##获取性能数据，不需要修改
#吞吐量
ActualFPS=${FPS}
#单迭代训练时长
TrainingTime=`awk 'BEGIN{printf "%.3f\n", '${time}'}'`

#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要模型审视修改
grep "mmseg - INFO - Iter " ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_${ASCEND_DEVICE_ID}_8p.log | awk -F "loss: " '{print $2}' >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt
#最后一个迭代loss值，不需要修改
ActualLoss=`awk 'END {print}' ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt`

#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" > ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainAccuracy = ${train_accuracy}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log