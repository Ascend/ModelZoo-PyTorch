#!/bin/bash

#current path, no revsion
cur_path=`pwd`

# 数据集路径,保持为空,不需要修改
data_path=""
RANK_SIZE=1
#设置默认日志级别,不需要修改
#export ASCEND_GLOBAL_LOG_LEVEL_ETP=3
#export ASCEND_GLOBAL_EVENT_ENABLE=1
export ENABLE_RUNTIME_V2=0
#基础参数，需要模型审视修改
#网络名称，同目录名称
Network="GaitSet"
#训练batch_size
batch_size=128
#训练步数
iters=40000

for para in $*
do
	if [[ $para == --data_path* ]];then
		data_path=`echo ${para#*=}`
    elif [[ $para == --iters* ]];then
        iters=`echo ${para#*=}`
	fi
done

#进入训练脚本目录，需要模型审视修改
cd $cur_path/../

#创建DeviceID输出目录，不需要修改
if [ -d ${cur_path}/output/${ASCEND_DEVICE_ID} ];then
    mkdir -p ${cur_path}/output/$ASCEND_DEVICE_ID/ckpt
else
    mkdir -p ${cur_path}/output/$ASCEND_DEVICE_ID/ckpt
fi

#训练开始时间，不需要修改
start_time=$(date +%s)
#执行训练脚本，传参需要审视修改
python3 train_main.py  --data_path=${data_path} \
                       --iters=${iters} > ${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &
wait
python3 test_main.py --iter=${iters} >> ${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &
wait

#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

#结果打印，不需要修改
echo "------------------ Final result ------------------"
sed -i "s|\r|\n|g" $cur_path/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log
#输出性能FPS，需要模型审视修改
FPS=`grep "FPS" ${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | awk -F "FPS" '{print$2}' | tr -d "(" | tr -d ")" | tr -d "',"|awk -F " " 'END{print$3}'`
#打印，不需要修改
echo "Final Performance images/sec : $FPS"

# #输出训练精度,需要模型审视修改
train_accuracy=`grep "NM:" $cur_path/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk -F "NM:" '{print$2}'| tr -d ","|head -n 1|awk -F " " '{print$1}'`

# #打印，不需要修改
# echo "Final Train Accuracy(top1): ${train_accuracy}"
# echo "Final Train Accuracy(top5): ${train_accuracy1}"
# echo "E2E Training Duration sec : $e2e_time"
#性能看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'perf'

##获取性能数据，不需要修改
#吞吐量
ActualFPS=${FPS}
#单迭代训练时长
TrainingTime=`awk 'BEGIN{printf "%.2f\n",'${BatchSize}'*1000/'${FPS}'}'`

#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
grep "Full_Loss" ${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log| awk -F'Full_Loss' '{print$2}'|awk -F' ' '{print$1}' > $cur_path/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt
#最后一个迭代loss值，不需要修改
ActualLoss=`awk 'END {print}' $cur_path/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt`
#编译耗时
# Make_Time=`grep -a 'TOTLE_TIME' ${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk 'END {print $3}'`

#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" > ${cur_path}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >> ${cur_path}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >> ${cur_path}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> ${cur_path}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> ${cur_path}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >> ${cur_path}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >> ${cur_path}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainAccuracy = ${train_accuracy}" >> ${cur_path}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> ${cur_path}/output/$ASCEND_DEVICE_ID/${CaseName}.log
#echo "Make_Time = None" >> ${cur_path}/output/$ASCEND_DEVICE_ID/${CaseName}.log