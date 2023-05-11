#!/usr/bin/env bash

source env_npu.sh 

# 网络名称，同目录名称
Network="Efficient-3DCNNs"
# 训练batch_size
batch_size=80


root_path=$1

#################创建日志输出目录，不需要修改#################
ASCEND_DEVICE_ID=0
if [ -d ${root_path}/results/${ASCEND_DEVICE_ID} ];then
    rm -rf ${root_path}/results/${ASCEND_DEVICE_ID}
    mkdir -p ${root_path}/results/$ASCEND_DEVICE_ID
else
    mkdir -p ${root_path}/results/$ASCEND_DEVICE_ID
fi

#################启动训练脚本#################
#训练开始时间，不需要修改
start_time=$(date +%s)




nohup python3 ../main.py \
    --root_path ${root_path} \
    --gpu_or_npu npu \
    --use_prof 1 \
    --use_apex 1 \
    --device_lists 0 \
    --distributed 0 \
    --n_classes 101 \
    --n_finetune_classes 101 \
	  --learning_rate 0.01 \
	  --droupout_rate 0.9 \
	  --n_epochs 2 \
	  --batch_size ${batch_size} \
	  --n_threads 16 \
	  --ft_portion complete \
	  > ${root_path}/results/${ASCEND_DEVICE_ID}/npu_train_performance_1p.log 2>&1 &


wait

##################获取训练数据################
# 训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

# 结果打印，不需要修改
echo "------------------ Final result ------------------"
# 输出性能FPS，需要模型审视修改
FPS=`grep 'fps' ${root_path}/results/${ASCEND_DEVICE_ID}/npu_train_performance_1p.log| tail -n 1 |awk '{print $6}'|awk 'END {print}'`
FPS=${FPS%,*}
# 打印，不需要修改
echo "Final Performance images/sec : $FPS"

# 输出训练精度,需要模型审视修改
top1_acc=`grep 'test top1 acc' ${root_path}/results/${ASCEND_DEVICE_ID}/npu_train_performance_1p.log|awk '{print $4}'|awk 'END {print}'`
top1_acc=`echo ${top1_acc%,*}`
# 打印，不需要修改
echo "Final Train Accuracy : ${top1_acc}"
echo "E2E Training Duration sec : $e2e_time"

# 性能看护结果汇总
# 训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'acc'

# 获取性能数据，不需要修改
# 吞吐量
ActualFPS=${FPS}
# 单迭代训练时长
TrainingTime=`awk 'BEGIN{printf "%.2f\n", '${batch_size}'*1000/'${FPS}'}'`

# 从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
grep 'date' ${root_path}/results/${ASCEND_DEVICE_ID}/npu_train_performance_1p.log|awk '{print $8}' >> ${root_path}/results/${ASCEND_DEVICE_ID}/train_${CaseName}_loss.txt
# 删除loss值后逗号
sed -i 's/,//g' ${root_path}/results/${ASCEND_DEVICE_ID}/train_${CaseName}_loss.txt

# 最后一个迭代loss值，不需要修改
ActualLoss=`awk 'END {print}'  ${root_path}/results/${ASCEND_DEVICE_ID}/train_${CaseName}_loss.txt`

# 关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" >  ${root_path}/results/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >>  ${root_path}/results/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >>  ${root_path}/results/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >>  ${root_path}/results/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >>  ${root_path}/results/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >>  ${root_path}/results/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >>  ${root_path}/results/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainAccuracy = ${top1_acc}" >> ${root_path}/results/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >>  ${root_path}/results/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >>  ${root_path}/results/$ASCEND_DEVICE_ID/${CaseName}.log
