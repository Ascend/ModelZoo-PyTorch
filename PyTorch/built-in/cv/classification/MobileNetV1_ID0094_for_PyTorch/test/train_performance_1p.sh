#!/bin/bash

###############指定训练脚本执行路径###############
# cd到与test文件夹同层级目录下执行脚本，提高兼容性；test_path_dir为包含test文件夹的路径
cur_path=`pwd`
cur_path_last_dirname=${cur_path##*/}
if [ x"${cur_path_last_dirname}" == x"test" ];then
    test_path_dir=${cur_path}
    cd ..
    cur_path=`pwd`
else
    test_path_dir=${cur_path}/test
fi

#集合通信参数,不需要修改

export RANK_SIZE=1
export JOB_ID=10087
RANK_ID_START=0

RANK_SIZE=1
# 数据集路径,保持为空,不需要修改
data_path=""
# 指定训练所使用的npu device卡id
device_id=0

#设置默认日志级别,不需要修改
export ASCEND_GLOBAL_LOG_LEVEL=3

#基础参数，需要模型审视修改
#网络名称，同目录名称
Network="MobileNetV1_ID0094_for_PyTorch"
#训练epoch
train_epochs=1
#训练batch_size
batch_size=256
#训练step
train_steps=`expr 1281167 / ${batch_size}`
#学习率
learning_rate=0.1

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
    elif [[ $para == --device_id* ]];then
      device_id=`echo ${para#*=}`
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
# 校验是否指定了device_id,分动态分配device_id与手动指定device_id,此处不需要修改
if [ $ASCEND_DEVICE_ID ];then
    echo "device id is ${ASCEND_DEVICE_ID}"
elif [ ${device_id} ];then
    export ASCEND_DEVICE_ID=${device_id}
    echo "device id is ${ASCEND_DEVICE_ID}"
else
    "[Error] device id must be config"
    exit 1
fi
#设置环境变量，不需要修改
echo "Device ID: $ASCEND_DEVICE_ID"
export RANK_ID=$RANK_ID

if [ -d $test_path_dir/output ];then
   rm -rf $test_path_dir/output/*
   mkdir -p $test_path_dir/output/$ASCEND_DEVICE_ID
else
   mkdir -p $test_path_dir/output/$ASCEND_DEVICE_ID
fi
wait

#非平台场景时source 环境变量
check_etp_flag=`env | grep etp_running_flag`
etp_flag=`echo ${check_etp_flag#*=}`
if [ x"${etp_flag}" != x"true" ];then
    source  ${test_path_dir}/env_npu.sh
fi

#训练开始时间，不需要修改
start_time=$(date +%s)

nohup python3 ${cur_path}/main.py \
    --arch=mobilenet \
    --data=${data_path} \
    --batch_size=${batch_size} \
    --learning-rate=${learning_rate} \
    --epochs=$train_epochs \
    --device_id=${device_id} \
    --max_steps=2000 \
    --loss-scale-value=1024 \
    --apex \
    --apex-opt-level='O2' \
    --workers=128 \
    --print-freq=10  > $test_path_dir/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log 2>&1 &
wait
#    --max_steps=2000 \

#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
FPS=`grep -a 'img/s'  $test_path_dir/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk '{print $9}'|awk 'END {print}'`
#打印，不需要修改
echo "Final Performance images/sec : $FPS"

#获取编译时间
CompileTime=`grep "Epoch" $test_path_dir/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | head -n 1|awk -F " " '{print $4}' | awk '{sum+=$1} END {print"",sum}' |sed s/[[:space:]]//g`

#输出训练精度,需要模型审视修改
train_accuracy=`grep -a '* Prec@1' $test_path_dir/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk 'END {print}'|awk '{print $3}'`

#打印，不需要修改
echo "Final Train Accuracy : ${train_accuracy}"
echo "E2E Training Duration sec : $e2e_time"

#性能看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'perf'

#获取性能数据，不需要修改
#吞吐量
ActualFPS=${FPS}
#单迭代训练时长
TrainingTime=`awk 'BEGIN{printf "%.2f\n", '${batch_size}'*1000/'${FPS}'}'`

#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
grep Epoch: $test_path_dir/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log| awk '{print $14}' > $test_path_dir/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt

#最后一个迭代loss值，不需要修改
ActualLoss=`awk 'END {print}' $test_path_dir/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt`

#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" > $test_path_dir/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >> $test_path_dir/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >> $test_path_dir/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> $test_path_dir/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> $test_path_dir/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >> $test_path_dir/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >> $test_path_dir/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> $test_path_dir/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> $test_path_dir/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CompileTime = ${CompileTime}" >> $test_path_dir/output/$ASCEND_DEVICE_ID/${CaseName}.log