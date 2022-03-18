#!/bin/bash

#当前路径,不需要修改
cur_path=`pwd`

#集合通信参数,不需要修改
export RANK_SIZE=8
RANK_ID_START=0

# 数据集路径,保持为空,不需要修改
data_path=""

#网络名称,同目录名称,需要模型审视修改
Network="FaceNet_ID0434_for_PyTorch"

#训练epoch
train_epochs=8

#训练batch_size,,需要模型审视修改
batch_size=4096

#参数校验，不需要修改
for para in $*
do
    if [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    fi
done

#校验是否传入data_path,不需要修改
if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
    exit 1
fi

#训练开始时间，不需要修改
start_time=$(date +%s)

#进入训练脚本目录，需要模型审视修改
cd $cur_path/../

#设置环境变量，不需要修改
RANK_ID=0
ASCEND_DEVICE_ID=0
echo "Device ID: $RANK_ID"
export RANK_ID=$RANK_ID
export ASCEND_DEVICE_ID=$RANK_ID
ASCEND_DEVICE_ID=$RANK_ID


if [ -d ${cur_path}/output/${ASCEND_DEVICE_ID} ];then
    rm -rf ${cur_path}/output/${ASCEND_DEVICE_ID}
    mkdir -p ${cur_path}/output/$ASCEND_DEVICE_ID/ckpt
else
    mkdir -p ${cur_path}/output/$ASCEND_DEVICE_ID/ckpt
fi

# 绑核，不需要的绑核的模型删除，需要的模型审视修改
#let a=RANK_ID*12
#let b=RANK_ID+1
#let c=b*12-1

#执行训练脚本，以下传参不需要修改，其他需要模型审视修改
rm -f nohup.out

KERNEL_NUM=$(($(nproc)/8))
for((RANK_ID=0;RANK_ID<RANK_SIZE;RANK_ID++))
do
    export RANK_ID=$RANK_ID

    if [ $(uname -m) = "aarch64" ]
    then
        PID_START=$((KERNEL_NUM * RANK_ID))
        PID_END=$((PID_START + KERNEL_NUM - 1))
        taskset -c $PID_START-$PID_END python3.7 fine_tune_new_8p.py \
            --seed 12345 \
            --amp_cfg \
            --opt_level O2 \
            --loss_scale_value 1024 \
            --device_list '0,1,2,3,4,5,6,7' \
            --batch_size $batch_size \
            --epochs $train_epochs \
            --epochs_per_save 5 \
            --lr 0.005 \
            --workers 64 \
            --data_dir $data_path \
            --addr=$(hostname -I |awk '{print $1}') \
            --rank 0 \
            --dist_url 'tcp://127.0.0.1:50000' \
            --dist_backend 'hccl' \
            --multiprocessing_distributed \
            --world_size 1 > ${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &
    else
        python3.7 fine_tune_new_8p.py \
            --seed 12345 \
            --amp_cfg \
            --opt_level O2 \
            --loss_scale_value 1024 \
            --device_list '0,1,2,3,4,5,6,7' \
            --batch_size $batch_size \
            --epochs $train_epochs \
            --epochs_per_save 5 \
            --lr 0.005 \
            --workers 64 \
            --data_dir $data_path \
            --addr=$(hostname -I |awk '{print $1}') \
            --rank 0 \
            --dist_url 'tcp://127.0.0.1:50000' \
            --dist_backend 'hccl' \
            --multiprocessing_distributed \
            --world_size 1 > ${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &
    fi
done
wait


#8p情况下仅0卡(主节点)有完整日志,因此后续日志提取仅涉及0卡
ASCEND_DEVICE_ID=0

#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
FPS=`grep Train  $cur_path/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|grep -v '1/'|grep -v '2/' |grep -v '3/'|awk -F 'fps:' '{print $2}'|awk '{print $1}'|awk '{sum+=$1} END {print  sum/NR*8}'`
#打印，不需要修改
echo "Final Performance images/sec : $FPS"

#输出训练精度,需要模型审视修改
train_accuracy=`grep -a 'Train' $cur_path/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log| awk -F " " '{print $12}'|awk 'END {print}'|sed 's/.$//'`
#打印，不需要修改
echo "Final Train Accuracy : ${train_accuracy}"
echo "E2E Training Duration sec : $e2e_time"

#稳定性精度看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'acc'

##获取性能数据，不需要修改
#吞吐量
ActualFPS=${FPS}
#单迭代训练时长
TrainingTime=`awk 'BEGIN{printf "%.2f\n", '${batch_size}'/'${FPS}'}'`

#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要模型审视修改
grep -a Train $cur_path/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|grep loss|awk -F "loss: " '{print $NF}' | awk -F " " '{print $1}' >  $cur_path/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt
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
echo "TrainAccuracy = ${train_accuracy}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log