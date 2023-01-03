#!/bin/bash

#网络名称,同目录名称,需要模型审视修改
Network="yolov4"

cur_path=`pwd`
model_name=yolov4-pacsp
batch_size=128

# 参数校验，data_path为必传参数，其他参数的增删由模型自身决定；此处新增参数需在上面有定义并赋值
for para in $*
do
    if [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    fi
done

# 校验是否传入data_path,不需要修改
if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
    exit 1
fi

mkdir -p ./data/VOC/images
ln -snf $data_path ./data/VOC/images/VOCdevkit
python3 convert_voc.py
wait

# 校验是否指定了device_id,分动态分配device_id与手动指定device_id,此处不需要修改
ASCEND_DEVICE_ID=0
echo "device id is ${ASCEND_DEVICE_ID}"

source ${cur_path}/test/env_npu.sh

#训练开始时间，不需要修改
start_time=$(date +%s)
echo "start_time: ${start_time}"

if [ -d ${cur_path}/test/output/${ASCEND_DEVICE_ID} ];
then
	rm -rf ${cur_path}/test/output/${ASCEND_DEVICE_ID}
	mkdir -p ${cur_path}/test/output/${ASCEND_DEVICE_ID}
else
	mkdir -p ${cur_path}/test/output/${ASCEND_DEVICE_ID}
fi

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
export WORLD_SIZE=8
KERNEL_NUM=$(($(nproc)/8))
for i in $(seq 0 7)
do 
export RANK=$i
let p_start=$((KERNEL_NUM * i))
let p_end=$((p_start + KERNEL_NUM - 1))
nohup taskset -c $p_start-$p_end python3 train.py \
    --workers 8 \
    --batch-size $batch_size \
    --data data/VOC.yaml \
    --img 640 640 \
    --cfg cfg/yolov4-pacsp.voc.cfg \
    --weights '' \
    --name $model_name \
    --local_rank $i \
    --hyp data/hyp.scratch.yaml > ${cur_path}/test/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &
done
wait

# #训练结束时间，不需要修改
end_time=$(date +%s)
echo "end_time: ${end_time}"
e2e_time=$(( $end_time - $start_time ))

#最后一个迭代FPS值
FPS=`grep -a 'FPS' ${cur_path}/test/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | awk -F " " '{print $4}' | sort -n | tail -5 | awk '{a+=$1} END {if (NR != 0) printf("%.3f",a/NR)}'`

#取acc值
acc=`grep -a 'all' ${cur_path}/test/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | awk 'END {print}' | awk -F " " '{print $6}'`

#打印，不需要修改
echo "ActualFPS : $FPS"
echo "ActualACC : $acc"
echo "E2E Training Duration sec : $e2e_time"

#稳定性精度看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${WORLD_SIZE}'p'_'acc'

##获取性能数据，不需要修改
#单迭代训练时长
TrainingTime=`awk 'BEGIN{printf "%.2f\n", '${batch_size}'*1000/'${FPS}'}'`

#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" > $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${FPS}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
