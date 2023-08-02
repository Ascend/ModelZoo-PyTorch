#!/bin/bash

#当前路径,不需要修改
cur_path=`pwd`

#集合通信参数,不需要修改

export RANK_SIZE=1
export JOB_ID=10087
RANK_ID_START=0

RANK_SIZE=1
# 数据集路径,保持为空,不需要修改
data_path=""

#设置默认日志级别,不需要修改
export ASCEND_GLOBAL_LOG_LEVEL=3

#基础参数，需要模型审视修改
#网络名称，同目录名称
Network="FFDNet_ID0970_for_PyTorch"
#训练epoch
train_epochs=1
#训练batch_size
batch_size=256
#训练step
#train_steps=`expr 1281167 / ${batch_size}`
#学习率
learning_rate=0.001

PREC=""
#维持参数，以下不需要修改
over_dump=False
data_dump_flag=False
data_dump_step="10"
profiling=False


#if [[ $1 == --help || $1 == --h ]];then
#   echo "usage:./train_performance_1p.sh --data_path=data_dir --batch_size=1024 --learning_rate=0.04"
#   exit 1
#fi
# 帮助信息，不需要修改
if [[ $1 == --help || $1 == -h ]];then
    echo"usage:./train_performance_1P.sh <args>"
    echo " "
    echo "parameter explain:
    --precision_mode         	precision mode(O0/O1/O2/O3)
    --over_dump		           	if or not over detection, default is False
    --data_dump_flag		    data dump flag, default is False
    --data_dump_step		    data dump step, default is 10
    --profiling		           	if or not profiling for performance debug, default is False
    --data_path		           	source data of training
    -h/--help		            show help message
    "
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
    elif [[ $para == --precision_mode* ]];then
        precision_mode=`echo ${para#*=}`
    elif [[ $para == --batch_size* ]];then
      batch_size=`echo ${para#*=}`
    elif [[ $para == --learning_rate* ]];then
      learning_rate=`echo ${para#*=}`
    fi
done

#PREC=""
#if [[ $precision_mode == "amp" ]];then
#  PREC="--apex"
#fi

#校验是否传入data_path,不需要修改
if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
    exit 1
fi
   
cd $cur_path
#设置环境变量，不需要修改
echo "Device ID: $ASCEND_DEVICE_ID"
export RANK_ID=$RANK_ID

if [ -d $cur_path/output ];then
   rm -rf $cur_path/output/*
   mkdir -p $cur_path/output/$ASCEND_DEVICE_ID
else
   mkdir -p $cur_path/output/$ASCEND_DEVICE_ID
fi
wait

cd ..

#export NPU_CALCULATE_DEVICE=$ASCEND_DEVICE_ID
#训练开始时间，不需要修改
start_time=$(date +%s)

nohup python3 ffdnet.py  \
        --is_train \
        --use_gpu \
        --epoches ${train_epochs} \
        --batch_size ${batch_size} \
		${PREC} \
        --train_path ${data_path}/datatrain/ > $cur_path/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log 2>&1 &

wait

#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
FPS=`grep -a 'fps'  $cur_path/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk '{print $18}'|awk '{sum+=$1} END {print sum/NR}'`
#打印，不需要修改
echo "Final Performance images/sec : $FPS"

#输出训练精度,需要模型审视修改
train_accuracy=`grep 'all' $cur_path/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk '{print $5}'`
#打印，不需要修改
echo "Final Train Accuracy : ${train_accuracy}"
echo "E2E Training Duration sec : $e2e_time"
#获取编译时间
CompileFps=`grep "fps:" $cur_path/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log| head -1 | awk '{print $18}' |sed s/[[:space:]]//g`
CompileTime=`awk 'BEGIN{printf "%.2f\n",'${batch_size}'/'${CompileFps}'}'`
#性能看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'perf'

##获取性能数据，不需要修改
#吞吐量
ActualFPS=${FPS}
#单迭代训练时长
TrainingTime=`awk 'BEGIN{printf "%.2f\n", '${batch_size}'*1000/'${FPS}'}'`

#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
grep 'Batch' $cur_path/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log| awk '{print $14}'|awk -F "," '{print $1}' >> $cur_path/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt

#最后一个迭代loss值，不需要修改
ActualLoss=`awk 'END {print}' $cur_path/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt`

#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" > $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CompileTime = ${CompileTime}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log