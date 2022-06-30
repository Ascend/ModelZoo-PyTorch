#!/bin/bash

#当前路径,不需要修改
cur_path=`pwd`

#集合通信参数,不需要修改

export RANK_SIZE=8
export JOB_ID=10087
RANK_ID_START=0

# 数据集路径,保持为空,不需要修改
data_path="/npu/traindata/imagenet_pytorch/"

#设置默认日志级别,不需要修改
export ASCEND_GLOBAL_LOG_LEVEL=3

#基础参数，需要模型审视修改
#网络名称，同目录名称
Network="Wide_resnet50_2_ID0397_for_PyTorch"
#训练epoch
train_epochs=100
#训练batch_size
batch_size=32
#训练step
train_steps=`expr 1281167 / ${batch_size}`
#学习率
learning_rate=0.1

#维测参数，precision_mode需要模型审视修改
PREC=""

#维测参数，precision_mode需要模型审视修改
precision_mode="allow_mix_precision"
#维持参数，以下不需要修改
over_dump=False
data_dump_flag=False
data_dump_step="10"
profiling=False

if [[ $1 == --help || $1 == --h ]];then
   echo "usage:./train_performance_1p.sh --data_path=data_dir --batch_size=1024 --learning_rate=0.04
          --precision_mode         precision mode(O0/O1/O2/O3)
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
    elif [[ $para == --precision_mode* ]];then
        precision_mode=`echo ${para#*=}`
    elif [[ $para == --batch_size* ]];then
      batch_size=`echo ${para#*=}`
    elif [[ $para == --learning_rate* ]];then
      learning_rate=`echo ${para#*=}`
    fi
done

#校验是否传入data_path,不需要修改
if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
    exit 1
fi

cd $cur_path
for((RANK_ID=$RANK_ID_START;RANK_ID<$((RANK_SIZE+RANK_ID_START));RANK_ID++));
do
    #设置环境变量，不需要修改
    export ASCEND_DEVICE_ID=$RANK_ID
    echo "Device ID: $ASCEND_DEVICE_ID"
    export WORLD_SIZE=$RANK_SIZE
    export LOCAL_RANK=$RANK_ID
    export RANK_ID=$RANK_ID
    export RANK=$RANK_ID
    export MASTER_ADDR=127.0.0.1
    export MASTER_PORT=29688

    if [ -d $cur_path/output/$ASCEND_DEVICE_ID ];then
       rm -rf $cur_path/output/*
       mkdir -p $cur_path/output/$ASCEND_DEVICE_ID
    else
       mkdir -p $cur_path/output/$ASCEND_DEVICE_ID
    fi


    #训练开始时间，不需要修改
    start_time=$(date +%s)

    nohup python3.7 $cur_path/../train.py  \
            --model wide_resnet50_2 \
            --epochs ${train_epochs} \
            --device $ASCEND_DEVICE_ID \
            --data-path ${data_path} \
            --batch-size ${batch_size} $PREC \
            --lr ${learning_rate} \
            $PREC \
            --momentum 0.9 \
            --apex \
            --weight-decay 1e-4 \
            --print-freq 10 > $cur_path/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log 2>&1 &
done
wait

#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))
ASCEND_DEVICE_ID=0
#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
FPS=`grep -a 'img/s'  $cur_path/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log | awk -F "img/s" '{print $2}'|awk '{print$2}' |awk '{sum+=$1} END {print sum/NR}'|sed s/[[:space:]]//g`
#打印，不需要修改
echo "Final Performance images/sec : $FPS"

#输出训练精度,需要模型审视修改
train_accuracy=`grep -a '* Acc@1' $cur_path/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk '{print $3}'| awk 'BEGIN {max = 0} {if ($1+0>max+0) max=$1 fi} END {print max}'`

#打印，不需要修改
echo "Final Train Accuracy : ${train_accuracy}"
echo "E2E Training Duration sec : $e2e_time"

#性能看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'acc'

##获取性能数据，不需要修改
#吞吐量
ActualFPS=${FPS}
#单迭代训练时长
TrainingTime=`awk 'BEGIN{printf "%.2f\n", '${batch_size}'*1000/'${FPS}'}'`

#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
grep img/s $cur_path/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log | awk -F "img/s" '{print$2}' | awk '{print$4}' > $cur_path/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt


#最后一个迭代loss值，不需要修改
ActualLoss=`awk '{if($0!="")print}' $cur_path/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt |awk 'END {print}'`

#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" > $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainAccuracy = ${train_accuracy}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log