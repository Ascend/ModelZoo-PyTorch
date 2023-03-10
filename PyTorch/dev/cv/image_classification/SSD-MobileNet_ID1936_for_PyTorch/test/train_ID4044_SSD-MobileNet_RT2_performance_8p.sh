#!/bin/bash

#当前路径,不需要修改
cur_path=`pwd`
#集合通信参数,不需要修改
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29688
export RANK_SIZE=8
export JOB_ID=10087

# 数据集路径,保持为空,不需要修改
data_path=""

#网络名称,同目录名称,需要模型审视修改
Network="SSD-MobileNet_RT2_ID4044_for_PyTorch"

#训练batch_size,,需要模型审视修改
batch_size=64

#参数校验，不需要修改
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
#校验是否传入data_path,不需要修改
if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
    exit 1
fi


if [ -d $data_path/VOCdevkit ];then
	echo "NO NEED TARZXVF"
else
	tar -zxvf $data_path/VOCdevkit.tar.gz -C $data_path/
fi
wait


#训练开始时间，不需要修改

#进入训练脚本目录，需要模型审视修改
cd $cur_path/../

sed -i "s|: 20|: 1|g" ${cur_path}/../config.json
#修改数据集路径
sed -i "s|"1111"|"$data_path/VOCdevkit/VOC2007"|g" ${cur_path}/../config.json
sed -i "s|"2222"|"$data_path/VOCdevkit/VOC2012"|g" ${cur_path}/../config.json

start_time=$(date +%s)
RANK_ID_START=0
for((RANK_ID=$RANK_ID_START;RANK_ID<$((RANK_SIZE+RANK_ID_START));RANK_ID++));
do
    #设置环境变量，不需要修改
    echo "Device ID: $RANK_ID"
    export RANK_ID=$RANK_ID
    export ASCEND_DEVICE_ID=$RANK_ID
    export NPU_CALCULATE_DEVICE=$ASCEND_DEVICE_ID

    #创建DeviceID输出目录，不需要修改
    if [ -d ${cur_path}/output/${ASCEND_DEVICE_ID} ];then
        rm -rf ${cur_path}/output/${ASCEND_DEVICE_ID}
        mkdir -p ${cur_path}/output/$ASCEND_DEVICE_ID/ckpt
    else
        mkdir -p ${cur_path}/output/$ASCEND_DEVICE_ID/ckpt
    fi

    #执行训练脚本，以下传参不需要修改，其他需要模型审视修改
    nohup python3 train.py config.json > ${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &
done
wait

#conda deactivate
#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))
ASCEND_DEVICE_ID=0
sed -i "s|"$data_path/VOCdevkit/VOC2007"|"1111"|g" ${cur_path}/../config.json
sed -i "s|"$data_path/VOCdevkit/VOC2012"|"2222"|g" ${cur_path}/../config.json
sed -i "s|: 8e-3|: 1e-3|g" ${cur_path}/../config.json
#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
FPS=`grep "FPS = " $cur_path/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk -F "FPS = " '{print $2}'|awk -F ", step_time" '{print $1}' |tail -n+2 |awk '{sum+=$1} END {print"",sum/NR}'|sed s/[[:space:]]//g`
#打印，不需要修改
echo "Final Performance images/sec : $FPS"
echo "E2E Training Duration sec : $e2e_time"

#输出训练精度,需要模型审视修改
#train_accuracy=`grep "acc = " $cur_path/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|grep -v mlp_log|awk 'END {print $5}'| sed 's/,//g' |cut -c 1-5`
#打印，不需要修改
#echo "Final Train Accuracy : ${train_accuracy}"

#稳定性精度看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'perf'

##获取性能数据
#吞吐量，不需要修改
ActualFPS=${FPS}
#单迭代训练时长，不需要修改
TrainingTime=`awk 'BEGIN{printf "%.2f\n",'${BatchSize}'*1000/'${FPS}'}'`
#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
grep "Batch Time" $cur_path/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk -F "Loss " '{print $2}'|awk -F " " '{print$1}' > $cur_path/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt

#最后一个迭代loss值，不需要修改
ActualLoss=`awk 'END {print}' $cur_path/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt`

#输出训练精度
tail -5 $cur_path/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt|awk '{sum+=$1} END {print sum/NR}'

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