#!/bin/bash

################基础配置参数，需要模型审视修改##################
# 必选字段(必须在此处定义的参数): Network batch_size RANK_SIZE

cur_path=`pwd`
# 网络名称，同目录名称
Network="SOLOv2"
# 训练batch_size
batch_size=1
# 训练使用的npu卡数
export RANK_SIZE=1
data_path=""
MODEL="./test/work_dirs/solov2_release_r50_fpn_8gpu_1x/epoch_12.pth"
device_id=0

#参数校验，不需要修改
for para in $*
do
    if [[ $para == --device_id* ]];then
        device_id=`echo ${para#*=}`
    elif [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    elif [[ $para == --checkpoint* ]];then
        MODEL=`echo ${para#*=}`
    elif [[ $para == --batch_size* ]];then
        batch_size=`echo ${para#*=}`
    fi
done

#校验是否传入data_path,不需要修改
if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
    exit 1
fi
# 校验是否指定了device_id,分动态分配device_id与手动指定device_id,此处不需要修改

if [ $ASCEND_DEVICE_ID ];then
    echo "device id is ${ASCEND_DEVICE_ID}"
elif [ ${device_id} ];then
    export ASCEND_DEVICE_ID=${device_id}
    echo "device id is ${ASCEND_DEVICE_ID}"
else
    "[Error] device id must be config"
    exit 1
fi

#训练开始时间，不需要修改
start_time=$(date +%s)

#创建DeviceID输出目录，不需要修改
if [ -d ${cur_path}/output/${ASCEND_DEVICE_ID} ];then
    rm -rf ${cur_path}/output/${ASCEND_DEVICE_ID}
    mkdir -p ${cur_path}/output/$ASCEND_DEVICE_ID
else
    mkdir -p ${cur_path}/output/$ASCEND_DEVICE_ID
fi
# 非平台场景时source 环境变量

check_etp_flag=`env | grep etp_running_flag`
etp_flag=`echo ${check_etp_flag#*=}`
if [ x"${etp_flag}" != x"true" ];then
    source ${cur_path}/test/env_npu.sh
fi
nohup python3.7 tools/test_ins.py configs/solov2/solov2_r50_fpn_8gpu_1x.py  $MODEL --show --out  results_solo.pkl --eval segm \
      --batch_size=${batch_size} --gpu-ids ${device_id} --data_root=$data_path > ${cur_path}/output/${ASCEND_DEVICE_ID}/eval_${ASCEND_DEVICE_ID}.log 2>&1 &
wait

##################获取训练数据################
# 训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

# 输出训练精度,需要模型审视修改 # eval.log | awk -F ',' '{print $1}' | awk '{print $2}' | awk ' END {print}'
train_accuracy=`grep -a 'maxDets' $cur_path/output/${ASCEND_DEVICE_ID}/eval_${ASCEND_DEVICE_ID}.log|awk -F " " '{print $13}'|head -n 1`
# 打印，不需要修改
echo "Final Train Accuracy : ${train_accuracy}"
echo "E2E Training Duration sec : $e2e_time"

# 性能看护结果汇总
# 训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'eval'

# 关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" >  ${cur_path}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >>  ${cur_path}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >>  ${cur_path}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >>  ${cur_path}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >>  ${cur_path}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainAccuracy = ${train_accuracy}" >> ${cur_path}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >>  ${cur_path}/output/$ASCEND_DEVICE_ID/${CaseName}.log