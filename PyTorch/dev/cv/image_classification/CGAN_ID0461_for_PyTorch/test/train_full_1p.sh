#!/bin/bash

cur_path=`pwd`/../
#Batch Size
batch_size=128
#网络名称，同目录名称
Network="CGAN_ID0461_for_PyTorch"
#Device数量，单卡默认为1
RANK_SIZE=1
#训练epoch，可选
train_epochs=3
#训练step
train_steps=
#学习率
learning_rate=0.0002

#参数配置
data_path=""

if [[ $1 == --help || $1 == --h ]];then
        echo "usage:./train_performance_1p.sh "
        exit 1
fi

for para in $*
do
        if [[ $para == --data_path* ]];then
                data_path=`echo ${para#*=}`
        fi
done

if [[ $data_path  == "" ]];then
        echo "[Error] para \"data_path\" must be config"
        exit 1
fi
##############执行训练##########
cd $cur_path
if [ -d $cur_path/test/output ];then
        rm -rf $cur_path/test/output/*
        mkdir -p $cur_path/test/output/$ASCEND_DEVICE_ID
else
        mkdir -p $cur_path/test/output/$ASCEND_DEVICE_ID
fi
wait
#修改数据集参数
sed -i "s|.\/celeba|$data_path\/celeba|g" ${cur_path}config.yml
wait
start=$(date +%s)
nohup python3 main.py > $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log 2>&1
wait
end=$(date +%s)
e2etime=$(( $end - $start ))
#数据集参数回改
sed -i "s|$data_path\/celeba|.\/celeba|g" ${cur_path}config.yml
wait

#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
TrainingTime=`grep Iteration $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk -F 'time elapsed:' '{print $2}'|awk 'NR>3'| awk '{sum+=$1} END {print  sum/NR*1000/100}'`
#单迭代训练时长
FPS=`awk 'BEGIN{printf "%.2f\n",'1000'*'${batch_size}'/'${TrainingTime}'}'`
#打印，不需要修改
echo "Final Performance images/sec : $FPS"
echo "Final Training Duration sec : $e2etime"


#打印，不需要修改
echo "Final Train Accuracy : ${train_accuracy}"


#性能看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'acc'

##获取性能数据，不需要修改
#吞吐量
ActualFPS=${FPS}
#单迭代训练时长

#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
grep Iteration $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log |awk -F 'Gen Loss:' '{print $2}'|awk  -F ',' '{print $1}' >> $cur_path/test/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt

#最后一个迭代loss值，不需要修改
ActualLoss=`tail -3|awk '{sum+=$1} END {print sum/NR}'  $cur_path/test/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt`

#输出训练精度,需要模型审视修改
#train_accuracy=`grep "accuracy:" $cur_path/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk 'END {print $8}'`
train_accuracy=`tail -3|awk '{sum+=$1} END {print sum/NR}'  $cur_path/test/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt`

#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" > $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainAccuracy = ${train_accuracy}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2etime}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
