#!/bin/bash
cur_path=`pwd`/../
#export ASCEND_DEVICE_ID=0
cd $cur_path
data_path="./data"
#more_path1="./uncased_L-12_H-768_A-12"
#more_path2="./uncased_L-12_H-768_A-12"

if [[ $1 == --help || $1 == --h ]];then
   echo "usage:./train_performance_1p.sh --data_path=."
   exit 1
fi

for para in $*
do
   if [[ $para == --data_path* ]];then
      data_path=`echo ${para#*=}`
   elif [[ $para == --precision_mode* ]];then
      precision_mode=`echo ${para#*=}`
   fi
done

PREC=""
if [[ $precision_mode == "amp" ]];then
  PREC="--apex"
fi

sed -i "s|data/cifar10|$data_path|g"  train.py
  
cd $cur_path
rm -rf ./pretraining_output

if [ -d $cur_path/test/output ];then
   rm -rf $cur_path/test/output/*
   mkdir -p $cur_path/test/output/$ASCEND_DEVICE_ID
else
   mkdir -p $cur_path/test/output/$ASCEND_DEVICE_ID
fi
wait

start=$(date +%s)

nohup python3 train.py \
  --device_id $ASCEND_DEVICE_ID \
  --depth 16 \
  --width 8 \
  -epochs=200 \
  --extent 0  $PREC \
  --extra_params False \
  --mlp False > $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log 2>&1 &
wait
end=$(date +%s)
e2e_time=$(( $end - $start ))
cp -r $cur_path/pretraining_output $cur_path/test/output/$ASCEND_DEVICE_ID

step_sec=`grep -a 'Epoch:' $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk 'END {print $4}'`
average_perf=`awk 'BEGIN{printf "%.2f\n",'1000'/'$step_sec'}'`

echo "Final Performance ms/step : $average_perf"
echo "Final Training Duration sec : $e2e_time"

###下面字段用于冒烟看护
##定义网络基本信息
#网络名称，同目录名称
Network="GENet_ID0671_for_PyTorch"
#Device数量，单卡默认为1
RANK_SIZE=1
#BatchSize
BatchSize=128
#设备类型，自动获取，此处无需修改
DeviceType=`uname -m`
#用例名称，自动获取，此处无需修改
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'acc'

##获取性能数据
#吞吐量
ActualFPS=`grep Epoch: $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk -F "FPS" '{print$2}' | awk '{print$1}' | tail -n+3| awk '{sum+=$1} END {print"",sum/NR}'|sed s/[[:space:]]//g`
#单迭代训练时长
TrainingTime=`cat $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|grep "Epoch:"|awk -F " " '{print $4}' |awk '{sum+=$1} END {print"",sum/NR}'|sed s/[[:space:]]//g`

##获取Loss
#从train_$ASCEND_DEVICE_ID.log提取Loss到${CaseName}_loss.txt中，需要修改***匹配规则
grep "Epoch:" $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk -F "Loss" '{print$2}' | awk '{print$1}' > $cur_path/test/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt
#最后一个迭代Loss值
ActualLoss=`cat $cur_path/test/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt|awk 'END {print $1}'`
#输出训练精度,需要模型审视修改
train_accuracy_Error1=`grep "* Error@1" $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk -F "Error@1" '{print$2}' | awk '{print$1}' | awk 'NR==1{min=$1;next}{min=min<$1?min:$1}END{print min}'`
#train_accuracy_Error5=`cat $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|grep "Epoch:"|awk -F " " '{print $18}'|awk 'END {print $1}'`

#关键信息打印到CaseName.log中，此处无需修改
echo "Network = ${Network}" > $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainAccuracy = ${train_accuracy_Error1}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
#echo "TrainAccuracy_Error@5 = ${train_accuracy_Error5}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log