#!/bin/bash

################基础配置参数，需要模型审视修改##################
# 必选字段(必须在此处定义的参数): Network batch_size RANK_SIZE
# 网络名称，同目录名称
Network="Hourglass_for_PyTorch"
# 训练使用的npu卡数
export RANK_SIZE=8
# 训练batch_size
batch_size=32
checkpoint=./work_dirs/hourglass52_mpii_384x384/latest.pth
###############指定训练脚本执行路径###############
# cd到与test文件夹同层级目录下执行脚本，提高兼容性；test_path_dir为包含test文件夹的路径
cur_path=`pwd`
cur_path_last_dirname=${cur_path##*/}
if [ x"${cur_path_last_dirname}" == x"test" ];then
    test_path_dir=${cur_path}
    cd ..
    cur_path=`pwd`
else
    test_path_dir=${cur_path}/test
fi

#################创建日志输出目录，不需要修改#################
if [ -d ${test_path_dir}/output/${RANK_SIZE} ];then
    rm -rf ${test_path_dir}/output/${RANK_SIZE}
    mkdir -p ${test_path_dir}/output/$RANK_SIZE
else
    mkdir -p ${test_path_dir}/output/$RANK_SIZE
fi


#################启动训练脚本#################
#训练开始时间，不需要修改
start_time=$(date +%s)
# 非平台场景时source 环境变量
check_etp_flag=`env | grep etp_running_flag`
etp_flag=`echo ${check_etp_flag#*=}`
if [ x"${etp_flag}" != x"true" ];then
    source ${test_path_dir}/env.sh
fi

cd mmpose-master
./tools/dist_test.sh ./configs/top_down/hourglass/mpii/hourglass52_mpii_384x384.py ${checkpoint} ${NPUS:-${RANK_SIZE}} --eval PCKh > ${test_path_dir}/output/${RANK_SIZE}/train_${RANK_SIZE}.log 2>&1 &
wait
cd ..
#获取训练数据
#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

#结果打印，不需要修改
echo "------------------ Final result ------------------"

#输出训练精度,需要模型审视修改
train_accuracy=`grep -a '* Acc@1' ${test_path_dir}/output/${RANK_SIZE}/train_${RANK_SIZE}.log|awk 'END {print}'|awk -F "Acc@1" '{print $NF}'|awk -F " " '{print $1}'`
#打印，不需要修改
echo "Final Train Accuracy : ${train_accuracy}"
echo "E2E Training Duration sec : $e2e_time"

#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'eval'


#最后一个迭代loss值，不需要修改
ActualLoss=`grep Test ${test_path_dir}/output/$RANK_SIZE/train_${RANK_SIZE}.log |awk -F "Loss" '{print $NF}' | awk -F " " '{print $1}' | awk 'END {print}'`
#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" >  ${test_path_dir}/output/$RANK_SIZE/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >>  ${test_path_dir}/output/$RANK_SIZE/${CaseName}.log
echo "BatchSize = ${BatchSize}" >>  ${test_path_dir}/output/$RANK_SIZE/${CaseName}.log
echo "DeviceType = ${DeviceType}" >>  ${test_path_dir}/output/$RANK_SIZE/${CaseName}.log
echo "CaseName = ${CaseName}" >>  ${test_path_dir}/output/$RANK_SIZE/${CaseName}.log
echo "TrainAccuracy = ${train_accuracy}" >> ${test_path_dir}/output/$RANK_SIZE/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >>  ${test_path_dir}/output/$RANK_SIZE/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >>  ${test_path_dir}/output/$RANK_SIZE/${CaseName}.log