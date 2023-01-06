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
train_epochs=
#训练step
train_steps=
#学习率
learning_rate=1e-3

#参数配置
data_path=""
PREC=""
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
	        echo "[ERROR] para "precision_mode" must be config O1 or O2 or O3"
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




target_data="ID1182_CarPeting_Pytorch_RRN_DATA"


if [ -d "$data_path/../$target_data/ID1182_CarPeting_Pytorch_RRN" ];then
        python3 walk_xg.py --path $data_path/../$target_data/ID1182_CarPeting_Pytorch_RRN  --ch mm
else
        echo "FILE IS NOT EXIST!"
fi



if [ -d $data_path/../$target_data/ID1182_CarPeting_Pytorch_RRN ];then
    echo "NO NEED UNTAR"
else
        mkdir $data_path/../$target_data
	tar -zxvf $data_path/ID1182_CarPeting_Pytorch_RRN.tar.gz -C $data_path/../$target_data/
fi
wait


if [ -d "$data_path/../$target_data/ID1182_CarPeting_Pytorch_RRN" ];then
        python3 walk_xg.py --path $data_path/../$target_data/ID1182_CarPeting_Pytorch_RRN  
else
        echo "UNTAR FAIL"
fi




sed -i "s|pass|break|g" main.py
sed -i "s|/home/panj/data|$data_path/../$target_data/ID1182_CarPeting_Pytorch_RRN|g" ${cur_path}/main.py

export NPU_CALCULATE_DEVICE=$ASCEND_DEVICE_ID

start=$(date +%s)
nohup python3 main.py $PREC --batchsize 4 --nEpochs 1 > $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log 2>&1 &
wait
end=$(date +%s)
e2e_time=$(( $end - $start ))


#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
TrainingTime=0
perf=`grep -a "Timer" $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log |awk -F "Timer: " '{print $2}'|awk -F " sec" '{print$1}'|tail -n +3|awk '{sum+=$1} END {print"",sum/NR}'|sed s/[[:space:]]//g`

FPS=`awk -v x=$batch_size -v y=$perf 'BEGIN{printf "%.2f\n",x/y}'`
#打印，不需要修改
echo "Final Performance images/sec : $FPS"

#获取编译时间
CompileTime=`grep "Timer:" $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log | head -2 | awk -F "Timer:" '{print $2}' | awk -F " " '{print $1}' | awk '{sum+=$1} END {print"",sum}' |sed s/[[:space:]]//g`

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

grep "Timer" $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk -F "Loss: " '{print $2}'|awk -F " ||" '{print $1}' >$cur_path/test/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt

#精度值
#train_accuracy="?"

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
echo "E2ETrainingTime = ${e2e_time}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CompileTime = ${CompileTime}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log

sed -i "s|break|pass|g" main.py
sed -i "s|$data_path/../$target_data/ID1182_CarPeting_Pytorch_RRN|/home/panj/data|g" ${cur_path}/main.py


if [ -d "$data_path/../$target_data/ID1182_CarPeting_Pytorch_RRN" ];then
        python3 walk_xg.py --path $data_path/../$target_data/ID1182_CarPeting_Pytorch_RRN  --ch mm
else
        echo "FILE IS NOT EXIST!"
fi




