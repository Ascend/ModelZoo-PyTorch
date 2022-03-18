#!/bin/bash

################基础配置参数，需要模型审视修改##################
# 网络名称，同目录名称
Network="ResNeSt"
# 训练使用的npu卡数
export RANK_SIZE=8
# 设置忽视UserWarning
export PYTHONWARNINGS="ignore:semaphore_tracker:UserWarning"
# 配置文件路径
config_path=""
# loss scale
loss_scale=8
# optimizer level
opt_level=O1
# txt name
txt_name=log_${RANK_SIZE}P_acc_${opt_level}_${loss_scale}

# 参数校验
for para in $*
do
    if [[ $para == --config_path* ]];then
        config_path=`echo ${para#*=}`
    fi
done

# 校验必需参数config_path
if [ $config_path == "" ];then
    echo "[Error] para \"config_path\" must be confing"
    exit 1
fi

# 训练batch_size
batch_size=`cat ${config_path}|grep "BATCH_SIZE"|awk '{print $2}'`
if [ ! ${batch_size} ];then
  batch_size=64
  echo "The default value of batch size is 64"
fi


###############指定训练脚本执行路径###############
# cd到与test文件夹同层级目录下执行脚本，提高兼容性；test_path_dir为包含test文件夹的路径
cur_path=`pwd`
cur_path_last_diename=${cur_path##*/}
if [ x"${cur_path_last_diename}" == x"test" ];then
    test_path_dir=${cur_path}
    cd ..
    cur_path=`pwd`
else
    test_path_dir=${cur_path}/test
fi


#################创建日志输出目录，不需要修改#################
ASCEND_DEVICE_ID=0
if [ -d ${test_path_dir}/output/${ASCEND_DEVICE_ID} ];then
    rm -rf ${test_path_dir}/output/${ASCEND_DEVICE_ID}
    mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID
else
    mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID
fi


#################启动训练脚本#################
#训练开始时间，不需要修改
start_time=$(date +%s)
# 非平台场景时source 环境变量
check_etp_flag=`env | grep etp_running_flag`
etp_flag=`echo ${check_etp_flag#*=}`
if [ x"${etp_flag}" != x"true" ];then
    source ${test_path_dir}/env_npu.sh
fi

RANK_ID_START=0

for((RANK_ID=$RANK_ID_START;RANK_ID<$((RANK_ID_START+RANK_SIZE));RANK_ID++));
do
  KERNEL_NUM=$(($(nproc)/8))
  PID_START=$((KERNEL_NUM * RANK_ID))
  PID_END=$((PID_START + KERNEL_NUM - 1))

  nohup taskset -c $PID_START-$PID_END python3.7.5 -u ./train_npu.py  \
      --config-file "${config_path}"            \
      --device "npu"                            \
      --loss-scale ${loss_scale}                \
      --opt-level "${opt_level}"                \
      --outdir "./output_8p_full"               \
      --logtxt "${txt_name}.txt"                \
      --dist-backend "hccl"                     \
      --addr "127.0.0.1"                        \
      --port "29686"                            \
      --world-size 8                            \
      --rank ${RANK_ID}                         \
      --amp > ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &
done
wait


##################获取训练数据################
#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

#结果打印
echo "------------------ Final result ------------------"
#输出性能FPS
FPS=`grep -a 'FPS'  ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk -F " " '{print $7}'|awk 'END {print}'`
#打印
echo "Final Performance images/sec : $FPS"

#输出训练精度,需要模型审视修改
train_accuracy=`grep -a 'Top1'  ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk 'END {print}'|awk -F "Top1" '{print $NF}'|awk -F " " '{print $2}'`
#打印，不需要修改
echo "Final Train Accuracy : ${train_accuracy}"
echo "E2E Training Duration sec : $e2e_time"

#性能看护结果汇总
#训练用例信息
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'acc'

##获取性能数据
#吞吐量
ActualFPS=${FPS}
#单迭代训练时长
TrainingTime=`awk 'BEGIN{printf "%.2f\n", '${batch_size}'*1000/'${FPS}'}'`

#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中
grep Batch: ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk -F "Loss" '{print $NF}' | awk -F " " '{print $2}' >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt

#最后一个迭代loss值
ActualLoss=`awk 'END {print}'  ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt`

#关键信息打印到${CaseName}.log中
echo "Network = ${Network}" >  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
