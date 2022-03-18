#!/bin/bash

cur_path=`pwd`/../
#失败用例打屏
export ASCEND_SLOG_PRINT_TO_STDOUT=1

#基础参数，需要模型审视修改
#Batch Size
batch_size=100
#网络名称，同目录名称
Network="ICT_ID1179_for_PyTorch"
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

#sed -i "s|/media/yoo/Data/wider_face|$data_path|g" ${cur_path}/prepare_wider_data.py
#sed -i "s|1600|2|g" ${cur_path}/confs/wresnet28x2.yaml

export NPU_CALCULATE_DEVICE=$ASCEND_DEVICE_ID

start=$(date +%s)
nohup python3  main.py  --dataset cifar10  --num_labeled 400 --num_valid_samples 500 --root_dir experiments/ --data_dir $data_path --batch_size 100  --arch cnn13 --dropout 0.0 --mixup_consistency 100.0 --pseudo_label mean_teacher  --consistency_rampup_starts 0 --consistency_rampup_ends 100 --epochs 4  --lr_rampdown_epochs 450 --print_freq 200 --momentum 0.9 --lr 0.1 --ema_decay 0.999  --mixup_sup_alpha 1.0 --mixup_usup_alpha 1.0 > $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log 2>&1 &
wait
end=$(date +%s)
e2e_time=$(( $end - $start ))


#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
TrainingTime=0
perf=`grep Epoch $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk '{print $5}'|awk 'NR>2{print p}{p=$0} END{print p}'|awk -F "(" '{print $2}'|awk -F ")" '{print $1}'|awk '{sum+=$1} END {print"",sum/NR}'|sed s/[[:space:]]//g`

#perf=`sed -e 's//\n/g' $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log |grep "train"|grep "/125"|awk -F ", " '{print $2}'|awk -F "it/s" '{print $1}'|awk '{sum+=$1} END {print"",sum/NR}'|sed s/[[:space:]]//g`
FPS=`awk -v x=$batch_size -v y=$perf 'BEGIN{printf "%.2f\n",x*y}'`
#FPS=`awk 'BEGIN{printf "%.2f\n",$batch_size*$perf}'`
#打印，不需要修改
echo "Final Performance images/sec : $FPS"
echo "$perf"
#输出训练精度,需要模型审视修改
#打印，不需要修改
#echo "Final Train Accuracy : ${train_accuracy}"
echo "E2E Training Duration sec : $e2e_time"

#性能看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'perf'

##获取性能数据，不需要修改
#吞吐量
ActualFPS=${FPS}

#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
#grep total $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk '{print $9}' > $cur_path/test/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt

#sed -e 's//\n/g' $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log |grep train|awk -F "loss=" '{print $2}' |awk -F "," '{print $1}'  > $cur_path/test/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt
grep "Epoch" $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk -F "Class " '{print $2}'|awk -F " " '{print $1}' >$cur_path/test/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt

#精度值
train_accuracy=`grep Epoch $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log |awk '{print $17}'|awk 'NR==1{max=$1;next}{max=max>$1?max:$1}END{print max}'`
rm -rf $cur_path/test/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss_*

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

