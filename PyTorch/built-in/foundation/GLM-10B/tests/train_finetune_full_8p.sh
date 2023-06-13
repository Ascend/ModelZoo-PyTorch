#!/bin/bash

# 数据集路径,保持为空,不需要修改
data_path=""
#网络名称,同目录名称,需要模型审视修改
Network="GLM-10B_for_PyTorch"
#训练batch_size,,需要模型审视修改
batch_size=12
RANK_SIZE=16

cur_path=`pwd`

#参数校验，不需要修改
for para in $*
do
    if [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
	mkdir -p ${cur_path}/data
        ln -nsf ${data_path} ${cur_path}/data
    fi
done


#训练开始时间，不需要修改
start_time=$(date +%s)


#################创建日志输出目录，不需要修改#################
ASCEND_DEVICE_ID=0
test_path_dir=${cur_path}/tests
if [ -d ${test_path_dir}/output/${ASCEND_DEVICE_ID} ];then
    rm -rf ${test_path_dir}/output/${ASCEND_DEVICE_ID}
    mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID
else
    mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID
fi


#非平台场景时source 环境变量
check_etp_flag=`env | grep etp_running_flag`
etp_flag=`echo ${check_etp_flag#*=}`
if [ x"${etp_flag}" != x"true" ];then
    source  ${test_path_dir}/env_npu.sh
fi

#执行训练脚本，以下传参不需要修改，其他需要模型审视修改
bash scripts/ds_finetune_superglue.sh  config_tasks/model_blocklm_10B.sh config_tasks/task_copa.sh > ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &

wait

#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

#结果打印，不需要修改
echo "------------------ Final result ------------------"
echo "E2E Training Duration sec : $e2e_time"

#输出训练精度,需要模型审视修改
train_accuracy=`grep -a 'Found best accuracy' ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk -F "Found best accuracy " '{print $2}'|awk -F " " '{print $1}'|awk 'END {print}'`
#打印，不需要修改
echo "Final Train Accuracy : ${train_accuracy}"
#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'acc'

##获取性能数据，不需要修改
#吞吐量
ActualFPS=${FPS}


#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" > ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
