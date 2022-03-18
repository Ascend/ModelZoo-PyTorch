#!/bin/bash

cur_path=`pwd`/../
#失败用例打屏
export ASCEND_SLOG_PRINT_TO_STDOUT=0

#基础参数，需要模型审视修改
#Batch Size
batch_size=32
#网络名称，同目录名称
Network="ADACOS_ID1082_for_PyTorch"
#Device数量，单卡默认为1
RANK_SIZE=1
#训练epoch，可选
train_epochs=50
#训练step
train_steps=
#学习率
learning_rate=1e-3

#参数配置
data_path=""
PREC="--apex --apex-opt-level O2"

if [[ $1 == --help || $1 == --h ]];then
	echo "usage:./train_performance_1p.sh "
	exit 1
fi

for para in $*
do
	  if [[ $para == --data_path* ]];then
		  data_path=`echo ${para#*=}`
    elif [[ $para == --precision_mode* ]];then
        apex_opt_level=`echo ${para#*=}`
		    if [[ $apex_opt_level != "O1" ]] && [[ $apex_opt_level != "O2" ]] && [[ $apex_opt_level != "O3" ]]; then
			    echo "[ERROR] para \"precision_mode\" must be config O1 or O2 or O3"
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

sed -i "s|omniglot/omniglot|$data_path/omniglot|g" ${cur_path}/omniglot_train.py

export NPU_CALCULATE_DEVICE=$ASCEND_DEVICE_ID
cp $data_path/*.pth /root/.cache/torch/checkpoints/
start=$(date +%s)
nohup python3 omniglot_train.py $PREC --metric adacos --batch-size 32 --epochs 50 > $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log 2>&1 &
wait
end=$(date +%s)
e2e_time=$(( $end - $start ))



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

TrainingTime=0
grep "812/812" $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log > $cur_path/test/output/$ASCEND_DEVICE_ID/train_${CaseName}_perf_1.txt
sed -e 's/\r/\n/g' $cur_path/test/output/$ASCEND_DEVICE_ID/train_${CaseName}_perf_1.txt > $cur_path/test/output/$ASCEND_DEVICE_ID/train_${CaseName}_perf.txt
perf=`cat $cur_path/test/output/$ASCEND_DEVICE_ID/train_${CaseName}_perf.txt|grep "812/812"|awk -F "," '{print $2}'|awk -F "it" '{print $1}'|awk '{sum+=$1} END {print"",sum/NR}'|sed s/[[:space:]]//g`

FPS=`awk -v x=$batch_size -v y=$perf 'BEGIN{printf "%.2f\n",x*y}'`
TrainingTime=`awk 'BEGIN{printf "%.2f\n",'${batch_size}'*1000/'${FPS}'}'`
#打印，不需要修改
echo "Final Performance images/sec : $FPS"
rm -rf $cur_path/test/output/$ASCEND_DEVICE_ID/train_${CaseName}_perf_1.txt

ActualFPS=${FPS}

#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视

grep val_loss $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk -F "loss" '{print$2}' | awk '{print$1}' > $cur_path/test/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt
#精度值
train_accuracy=`grep "loss" $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk -F "acc@1" '{print$2}' | awk '{print$1}' | awk 'NR==1{max=$1;next}{max=max>$1?max:$1}END{print max}'`
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
echo "TrainAccuracy = ${train_accuracy}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log

