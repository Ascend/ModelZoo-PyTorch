#!/bin/bash
export ASCEND_SLOG_PRINT_TO_STDOUT=0

#集合通信参数,不需要修改
export HCCL_WHITELIST_DISABLE=1
RANK_SIZE=1
export RANK_SIZE=${RANK_SIZE}
export JOB_ID=10087
RANK_ID_START=0
# source env.sh
# 数据集路径,保持为空,不需要修改
data_path=""
device_id=0

#设置默认日志级别,不需要修改
# export ASCEND_GLOBAL_LOG_LEVEL_ETP=3

#基础参数，需要模型审视修改
#网络名称，同目录名称
Network="SSD_for_PyTorch"
#训练epoch
train_epochs=1
#训练batch_size
batch_size=24


#维持参数，以下不需要修改
over_dump=False
data_dump_flag=False
data_dump_step="10"
profiling=False


if [ $ASCEND_DEVICE_ID ];then
    echo "device id is ${ASCEND_DEVICE_ID}"
elif [ ${device_id} ];then
    export ASCEND_DEVICE_ID=${device_id}
    echo "device id is ${ASCEND_DEVICE_ID}"
else
    "[Error] device id must be config"
    exit 1
fi
 
cur_path=`pwd`
cur_path_last_dirname=${cur_path##*/}
if [ x"${cur_path_last_dirname}" == x"test" ];then
    test_path_dir=${cur_path}
    cd ..
    cur_path=`pwd`
else
    test_path_dir=${cur_path}/test
fi

#设置环境变量，不需要修改
echo "Device ID: $ASCEND_DEVICE_ID"
export RANK_ID=${$ASCEND_DEVICE_ID}

#################创建日志输出目录，不需要修改#################
if [ -d ${test_path_dir}/output/${ASCEND_DEVICE_ID} ];then
    rm -rf ${test_path_dir}/output/${ASCEND_DEVICE_ID}
    mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID
else
    mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID
fi
wait

#训练开始时间，不需要修改
start_time=$(date +%s)

# 非平台场景时source 环境变量
check_etp_flag=`env | grep etp_running_flag`
etp_flag=`echo ${check_etp_flag#*=}`
if [ x"${etp_flag}" != x"true" ];then
    source ${test_path_dir}/env_npu.sh
fi

# 绑核，不需要的绑核的模型删除，需要模型审视修改
RANK_ID_START=0
export WORLD_SIZE=${RANK_SIZE}
export RANK=${RANK_ID}
nohup python3.7 ./tools/train.py configs/ssd/ssdlite_mobilenetv2_scratch_600e_coco_1p.py \
    --seed 0 \
    --gpu-ids 0 \
    --opt-level O1  > ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log 2>&1 &
wait

#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))


#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
step_time=`grep "mmdet - INFO - Epoch" ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log | tail -n 10 | awk -F"time:" '{print $2}' | awk -F"," '{print $1}' | awk '{sum+=$1} END {print sum/NR}'`
FPS=`awk 'BEGIN{printf "%.2f\n", '$batch_size'/'$step_time'}'`

#打印，不需要修改
echo "Final Performance images/sec : $FPS"

#打印，不需要修改
echo "E2E Training Duration sec : $e2e_time"

#性能看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'perf'


ActualFPS=${FPS}

#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
echo "Network = ${Network}" > ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt

#最后一个迭代loss值，不需要修改
ActualLoss=`grep "mmdet - INFO - Epoch" ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log | tail -n 1 | awk -F"loss:" '{print $2}'`
echo "ActualLoss = ${ActualLoss}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt

#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" > ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "STEPTIME = ${step_time}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
