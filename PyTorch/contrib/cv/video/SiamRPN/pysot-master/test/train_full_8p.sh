#!/bin/bash

################基础配置参数，需要模型审视修改##################
# 必选字段(必须在此处定义的参数): Network batch_size RANK_SIZE
# 网络名称，同目录名称
Network="SimRPN_for_PyTorch"
# 训练使用的npu卡数
RANK_SIZE=8



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

for((RANK_ID=$RANK_ID_START;RANK_ID<$((RANK_SIZE+RANK_ID_START));RANK_ID++));
do
 KERNEL_NUM=$(($(nproc)/8))
 PID_START=$((KERNEL_NUM * RANK_ID))
 PID_END=$((PID_START + KERNEL_NUM - 1))

 export WORLD_SIZE=$RANK_SIZE


 nohup taskset -c $PID_START-$PID_END python3.7 -u ${test_path_dir}/../tools_8p/train.py \
  --cfg ${test_path_dir}/../experiments/siamrpn_r50_l234_dwxcorr_8gpu/config.yaml\
  --local_rank $RANK_ID > ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &

done

wait



python3.7  ${test_path_dir}/../tools_1p/test.py 	\
	--snapshot ${test_path_dir}/../snapshot_8p/checkpoint_e20.pth \
    --config ${test_path_dir}/../experiments/siamrpn_r50_l234_dwxcorr_8gpu/config.yaml \
	--datasetdir ${test_path_dir}/../../testing_dataset/VOT2016

wait


python3.7 ${test_path_dir}/../tools_1p/eval.py 	 \
	--tracker_path ${test_path_dir}/../results > ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_acc_${ASCEND_DEVICE_ID}.log 2>&1 &
wait


##################获取训练数据################
#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
FPS=`grep -a 'FPS' ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | awk -F " " '{print $2}'|awk 'END {print}'`
#打印，不需要修改
echo "Final Performance images/sec : $FPS"

ActualFPS=$FPS

#性能看护结果汇总
#训练用例信息，不需要修改
DeviceType=`uname -m`
CaseName=${Network}_${RANK_SIZE}'p'_'acc'

##获取性能数据，不需要修改


#提取Acc
Acc=`grep -a 'acc' ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_acc_${ASCEND_DEVICE_ID}.log | awk -F " " '{print $2}'|awk 'END {print}'`
Actualacc={$Acc}
echo "Fianl Acc $Acc"



#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" >  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "Actualacc = ${Actualacc}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log