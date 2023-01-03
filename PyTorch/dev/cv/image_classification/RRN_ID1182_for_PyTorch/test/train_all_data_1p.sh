#!/bin/bash

cur_path=`pwd`/../

#基础参数，需要模型审视修改
#Batch Size
batch_size=4
#网络名称，同目录名称
Network="RRN_ID1182_for_PyTorch"
#Device数量，单卡默认为1
RANK_SIZE=1
#训练epoch，可选
#train_epochs=70
train_epochs=47
#训练step
train_steps=
#学习率
learning_rate=1e-3

#参数配置
data_path=""

if [[ $1 == --help || $1 == --h ]];then
	echo "usage:./train_accormance_1p.sh "
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



target_data="ID1182_CarPeting_Pytorch_RRN_DATA"
if [ -d $data_path/../$target_data/ID1182_CarPeting_Pytorch_RRN ];then
    echo "NO NEED UNTAR"
else
    mkdir $data_path/../$target_data
    tar -zxvf $data_path/ID1182_CarPeting_Pytorch_RRN.tar.gz -C $data_path/../$target_data/
fi
wait





sed -i "s|/home/panj/data|$data_path/../$target_data/ID1182_CarPeting_Pytorch_RRN|g" ${cur_path}/main.py

export NPU_CALCULATE_DEVICE=$ASCEND_DEVICE_ID

start=$(date +%s)
nohup python3 main.py --batchsize 4 --nEpochs $train_epochs > $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log 2>&1 &
wait
end=$(date +%s)
e2e_time=$(( $end - $start ))


#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改

acc=`grep -a "Timer" $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log |awk -F "Timer: " '{print $2}'|awk -F " sec" '{print$1}'|awk '{sum+=$1} END {print"",sum/NR}'|sed s/[[:space:]]//g`

FPS=`awk -v x=$batch_size -v y=$acc 'BEGIN{printf "%.2f\n",x/y}'`


TrainingTime=0
#打印，不需要修改
echo "Final Performance images/sec : $FPS"

#输出训练精度,需要模型审视修改
#打印，不需要修改
#echo "Final Train Accuracy : ${train_accuracy}"
echo "E2E Training Duration sec : $e2e_time"

#性能看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'acc'

##获取性能数据，不需要修改
#吞吐量
ActualFPS=${FPS}

#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
#grep total $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk '{print $9}' > $cur_path/test/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt

grep "Timer" $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk -F "Loss: " '{print $2}'|awk -F " ||" '{print $1}' >$cur_path/test/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt


#精度值
train_accuracy="?"

#最后一个迭代loss值，不需要修改
ActualLoss=`awk 'END {print}' $cur_path/test/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt`

#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" > $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainAccuracy = ${train_accuracy}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log



if [ -d $data_path/ID1182_CarPeting_Pytorch_RRN ];then
	rm -rf $data_path/ID1182_CarPeting_Pytorch_RRN
else
        echo "NO NEED REMOVE"
fi      
wait  
