#!/bin/bash

#当前路径,不需要修改
cur_path=`pwd`

#集合通信参数,不需要修改

export RANK_SIZE=8
export JOB_ID=10087
RANK_ID_START=0
#source env.sh
RANK_SIZE=8
# 数据集路径,保持为空,不需要修改
data_path="/npu/traindata/imagenet_pytorch/"

#设置默认日志级别,不需要修改
export ASCEND_GLOBAL_LOG_LEVEL=3

#基础参数，需要模型审视修改
#网络名称，同目录名称
Network="MobileNetV3-Large_ID1784_for_PyTorch"
#训练epoch
train_epochs=1
#训练batch_size
batch_size=1024
#训练step
train_steps=`expr 1281167 / ${batch_size}`
#学习率
learning_rate=0.064


#维持参数，以下不需要修改
over_dump=False
data_dump_flag=False
data_dump_step="10"
profiling=False


if [[ $1 == --help || $1 == --h ]];then
   echo "usage:./train_performance_1p.sh --data_path=data_dir --batch_size=1024 --learning_rate=0.04"
   exit 1
fi

for para in $*
do
    if [[ $para == --data_path* ]];then
      data_path=`echo ${para#*=}`
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

#训练开始时间，不需要修改
start_time=$(date +%s)
RANK_ID_START=0
for((RANK_ID=$RANK_ID_START;RANK_ID<$((RANK_SIZE+RANK_ID_START));RANK_ID++));
do
    #设置环境变量，不需要修改
    echo "Device ID: $RANK_ID"
    export RANK_ID=$RANK_ID
    export ASCEND_DEVICE_ID=$RANK_ID
    ASCEND_DEVICE_ID=$RANK_ID

    #创建DeviceID输出目录，不需要修改
    if [ -d ${cur_path}/output/${ASCEND_DEVICE_ID} ];then
        rm -rf ${cur_path}/output/${ASCEND_DEVICE_ID}
        mkdir -p ${cur_path}/output/$ASCEND_DEVICE_ID/ckpt
    else
        mkdir -p ${cur_path}/output/$ASCEND_DEVICE_ID/ckpt
    fi

     # 绑核，不需要的绑核的模型删除，需要模型审视修改
    
	nohup python3 ${cur_path}/../main.py \
        --arch=mobilenet \
        --data=${data_path} \
        --batch_size=${batch_size} \
        --learning-rate=${learning_rate} \
        --epochs=$train_epochs \
        --apex \
        --apex-opt-level='O1' \
        --workers=128 \
        --print-freq=1 \
        --distributed \
		--lr-step-size=2 \
		--lr-gamma=0.973 \
		--wd=0.00001 \
        --world-size=1 \
		--max_steps=64 \
        --dist-rank=0 > $cur_path/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log 2>&1 &
 
done
wait
#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
FPS=`grep -a 'img/s'  $cur_path/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk '{print $9}'|awk 'END {print}'`
#打印，不需要修改
echo "Final Performance images/sec : $FPS"

#输出训练精度,需要模型审视修改
train_accuracy=`grep -a '* Prec@1' $cur_path/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk 'END {print}'|awk '{print $3}'`

#打印，不需要修改
echo "Final Train Accuracy : ${train_accuracy}"
echo "E2E Training Duration sec : $e2e_time"

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
grep Epoch: $cur_path/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log| awk '{print $14}' >> $cur_path/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt

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
