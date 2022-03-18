#!/bin/bash

cur_path=`pwd`/../
#失败用例打屏
#export ASCEND_SLOG_PRINT_TO_STDOUT=1

#基础参数，需要模型审视修改
#Batch Size
batch_size=256
#网络名称，同目录名称
Network="coral-cnn_ID1064_for_PyTorch"
#Device数量，单卡默认为1
RANK_SIZE=1
#训练epoch，可选
train_epochs=
#训练step
train_steps=
#学习率
learning_rate=1e-3

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




cp $data_path/coral-implementation-recipe.ipynb $cur_path/
cp $data_path/fig2-plot.ipynb  $cur_path/figure2/
cp $data_path/differences-at-a-glance.pdf  $cur_path/github-images/
cp $data_path/get-best-mae-rmse.ipynb  $cur_path/table1/
cp $data_path/afad.ipynb  $cur_path/table2/
cp $data_path/cacd.ipynb   $cur_path/table2/
cp $data_path/morph.ipynb   $cur_path/table2/

sed -i "s|./datasets|$data_path|g" ${cur_path}/model-code/afad-coral.py
sed -i "s|/shared_datasets/AFAD/orig/tarball|$data_path|g" ${cur_path}/model-code/afad-coral.py

export NPU_CALCULATE_DEVICE=$ASCEND_DEVICE_ID

start=$(date +%s)
nohup python3 model-code/afad-coral.py --seed 1 --cuda 0 --outpath afad-model1 --apex > $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log 2>&1 &
wait
end=$(date +%s)
e2e_time=$(( $end - $start ))


#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
FPS=`cat $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|grep "FPS"|awk -F "FPS:" '{print $2}'|cut -c 2-8|tail -n +2|awk '{sum+=$1} END {print sum/NR}'`
TrainingTime=`awk 'BEGIN{printf "%.2f\n",'${batch_size}'*1000/'${FPS}'}'`
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

grep "FPS" $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk -F "|" '{print $3}'|awk -F ":" '{print $2}' > $cur_path/test/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt

#精度值
#train_accuracy=`grep "loss" $cur_path/test/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss_2.txt|awk -F " " '{print $8}'|awk 'END {print}'`
train_accuracy=None
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
echo "TrainAccuracy = ${train_accuracy}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log

