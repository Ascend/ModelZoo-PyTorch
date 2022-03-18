#!/bin/bash

cur_path=`pwd`/../
#失败用例打屏
#export ASCEND_SLOG_PRINT_TO_STDOUT=1

#基础参数，需要模型审视修改
#Batch Size
batch_size=1
#网络名称，同目录名称
Network="POOLNET_ID0875_for_PyTorch"
#Device数量，单卡默认为1
RANK_SIZE=1
#训练epoch，可选
train_epochs=24
#训练step
train_steps=
#学习率
learning_rate=
PREC=""
#参数配置
data_path=""

if [[ $1 == --help || $1 == --h ]];then
	echo "usage:./train_performance_1p.sh"
	exit 1
fi

for para in $*
do
	if [[ $para == --data_path* ]];then
		data_path=`echo ${para#*=}`
    elif [[ $para == --precision_mode* ]];then
	    apex_opt_level=`echo ${para#*=}`
	    if [[ $apex_opt_level != "O1" ]] && [[ $apex_opt_level != "O2" ]] && [[ $apex_opt_level != "O3" ]]; then
		    echo "[Error] para \"precision_mode\" must be config O1 or O2 or O3"
		    exit 1
	    fi
	    PREC="--apex --apex-opt-level "$apex_opt_level
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

#参数数据集修改
sed -i "s|./dataset/|$data_path/|g" ${cur_path}/main.py
#sed -i "s|pass|break|g" ${cur_path}/solver.py
wait
#sed -i "s|self.num_epochs = 10|self.num_epochs = 1|g" ${cur_path}/models/TextRNN.p
export NPU_CALCULATE_DEVICE=$ASCEND_DEVICE_ID

start=$(date +%s)
nohup python3 main.py $PREC --arch resnet --train_root $data_path/DUTS/DUTS-TR --train_list $data_path/DUTS/DUTS-TR/train_pair.lst --epoch $train_epochs --batch_size $batch_size --epoch_save 1 > $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log 2>&1 &
wait
end=$(date +%s)
e2etime=$(( $end - $start ))

#参数数据集回改
sed -i "s|$data_path/|./dataset/|g" ${cur_path}/main.py
#sed -i "s|break|pass|g" ${cur_path}/solver.py
wait

echo "Final Training Duration sec : $e2etime"

#输出性能FPS，需要模型审视修改
FPS=`grep fps $cur_path/test/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk -F "fps :" '{print $2}'|tail -n+2|awk '{sum+=$1} END {print sum/NR}'|sed s/[[:space:]]//g`
#打印，不需要修改
cho "Final Performance images/sec : $FPS"

#输出训练精度,需要模型审视修改
#train_accuracy=`grep train_accuracy $cur_path/test/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk 'END {print $8}'|cut -c 1-5`
#打印，不需要修改
#echo "Final Train Accuracy : ${train_accuracy}"
echo "E2E Training Duration sec : $e2e_time"

#性能看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'acc'

#获取性能数据，不需要修改
#吞吐量
ActualFPS=${FPS}
#单迭代训练时长
TrainingTime=`awk 'BEGIN{printf "%.2f\n",'${batch_size}'*1000/'${ActualFPS}'}'`


#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
grep epoch $cur_path/test/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk -F "Sal :" '{print $2}'|awk '{print $1}'|sed s/[[:space:]]//g > $cur_path/test/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt
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
echo "ActualLoss = ${ActualLoss}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2etime}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
