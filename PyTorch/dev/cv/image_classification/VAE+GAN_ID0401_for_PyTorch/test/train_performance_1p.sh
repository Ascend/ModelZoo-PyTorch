#!/bin/bash

cur_path=`pwd`/../

#基础参数，需要模型审视修改
#Batch Size
batch_size=64
#网络名称，同目录名称
Network="VAE+GAN_ID0401_for_PyTorch"
#Device数量，单卡默认为1
RANK_SIZE=1
#训练epoch，可选
train_epochs=1
#训练step
train_steps=
#学习率
learning_rate=1e-3
#参数配置
data_path=""

PREC=""

if [[ $1 == --help || $1 == --h ]];then
	  echo"usage:./train_performance_1P.sh <args>"
    echo " "
    echo "parameter explain:
    --precision_mode         precision mode(O0/O1/O2/O3)
    --data_path		           source data of training
    -h/--help		             show help message
    "
    exit 1
fi

for para in $*
do
  if [[ $para == --precision_mode* ]];then
    apex_opt_level=`echo ${para#*=}`
    if [[ $apex_opt_level != "O1" ]] && [[ $apex_opt_level != "O2" ]] && [[ $apex_opt_level != "O3" ]]; then
        echo "[Error] para \"precision_mode\" must be config O1 or O2 or O3"
        exit 1
    fi
    PREC="--apex --apex-opt-level "$apex_opt_level
	elif [[ $para == --data_path* ]];then
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


export NPU_CALCULATE_DEVICE=$ASCEND_DEVICE_ID
sed -i "s|/content/drive/MyDATA|$data_path/../|g" $cur_path/dataloader.py
sed -i "s|epochs=25|epochs=$train_epochs|g" $cur_path/main.py
sed -i "s|pass|break|g" $cur_path/main.py
sed -i "s|if i % 50 == 0|if i % 10 == 0|g" $cur_path/main.py
wait
mkdir -p $data_path/raw
cp -rf $data_path/* $data_path/raw
RANK_ID=0
export RANK_ID=$RANK_ID
corenum=`cat /proc/cpuinfo|grep "processor" | wc -l`
let a=${RANK_ID}*${corenum}/${RANK_SIZE}
let b=${RAND_ID}+1
let c=${b}*${corenum}/${RANK_SIZE}-1

start=$(date +%s)
taskset -c $a-$c python3 main.py $PREC  > $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log 2>&1 &
wait
end=$(date +%s)
e2e_time=$(( $end - $start ))

sed -i "s|$data_path/../|/content/drive/MyDATA|g" $cur_path/dataloader.py
sed -i "s|epochs=$train_epochs|epochs=25|g" $cur_path/main.py
sed -i "s|break|pass|g" $cur_path/main.py
sed -i "s|if i % 10 == 0|if i % 50 == 0|g" $cur_path/main.py
wait

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

TrainingTime=0
FPS=`grep FPS: $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log |awk -F "FPS:" '{print$2}' | tail -n+2 | awk '{sum+=$1} END {print"",sum/NR}'|sed s/[[:space:]]//g`
TrainingTime=`awk 'BEGIN{printf "%.2f\n",'${batch_size}'*1000/'${FPS}'}'`

#打印，不需要修改
echo "Final Performance images/sec : $FPS"

ActualFPS=${FPS}

#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
grep FPS: $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log| awk -F "Loss_gan:" '{print$2}' | awk '{print$1}' > $cur_path/test/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt


#精度值
#train_accuracy=`grep "loss" $cur_path/test/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss_2.txt|awk -F " " '{print $8}'|awk 'END {print}'`

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
#echo "TrainAccuracy = ${train_accuracy}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
rm -rf $data_path/processed
rm -rf $data_path/raw