#!/bin/bash

##################基础配置参数，需要模型审视修改##################
# 必选字段(必须在此处定义的参数): Network batch_size RANK_SIZE
#网络名称，同目录名称
Network="ResNeXt101_32x8d"
#训练epoch
train_epochs=2
#训练batch_size
batch_size=1024
#学习率
learning_rate=0.4
# 训练使用的npu卡数
export RANK_SIZE=8
# 数据集路径,保持为空,不需要修改
data_path=""

#加载进程数
workers=$(nproc)

#维持参数，以下不需要修改
over_dump=False
data_dump_flag=False
data_dump_step="10"
profiling=False

# 帮助信息，不需要修改
if [[ $1 == --help || $1 == -h ]];then
    echo"usage:./train_performance_1P.sh <args>"
    echo " "
    echo "parameter explain:
    --precision_mode         precision mode(allow_fp32_to_fp16/force_fp16/must_keep_origin_dtype/allow_mix_precision)
    --over_dump		           if or not over detection, default is False
    --data_dump_flag		     data dump flag, default is False
    --data_dump_step		     data dump step, default is 10
    --profiling		           if or not profiling for performance debug, default is False
    --data_path		           source data of training
    -h/--help		             show help message
    "
    exit 1
fi

#参数校验，不需要修改
for para in $*
do
    if [[ $para == --workers* ]];then
        workers=`echo ${para#*=}`
    elif [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    fi
done

#校验是否传入data_path,不需要修改
if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
    exit 1
fi

##################指定训练脚本执行路径##################
# cd到与test文件同层级目录下执行脚本，提高兼容性；test_path_dir为包含test文件夹的路径
cur_path=`pwd`
cur_path_last_dirname=${cur_path##*/}
if [ x"${cur_path_last_dirname}" == x"test" ]; then
    test_path_dir=${cur_path}
    cd ..
    cur_path=`pwd`
else
    test_path_dir=${cur_path}/test
fi

# source 环境变量
ls -l ${test_path_dir}/env.sh 
source ${test_path_dir}/env.sh
##################指定训练脚本执行路径##################
# cd到与test文件同层级目录下执行脚本，提高兼容性；test_path_dir为包含test文件夹的路径
RANK_ID_START=0
RANK_SIZE=8
RANK_ID=0

for((RANK_ID=$RANK_ID_START;RANK_ID<$((RANK_SIZE+RANK_ID_START));RANK_ID++));
do
if [ -d ${test_path_dir}/output/$RANK_ID ];then
    rm -rf ${test_path_dir}/output/$RANK_ID
    mkdir -p ${test_path_dir}/output/$RANK_ID
else
    mkdir -p ${test_path_dir}/output/$RANK_ID
fi
done
##################启动训练脚本##################
#训练开始时间，不需要修改
start_time=$(date +%s)

RANK_ID_START=0
RANK_SIZE=8
RANK_ID=0


for((RANK_ID=${RANK_ID_START};RANK_ID<$((RANK_SIZE+RANK_ID_START));RANK_ID++));
do
KERNEL_NUM=$(($(nproc)/8))
PID_START=$((KERNEL_NUM*RANK_ID))
PID_END=$((PID_START+KERNEL_NUM-1))

nohup taskset -c $PID_START-$PID_END  python3.7 ./main.py \
	    ${data_path} \
      --addr=$(hostname -I |awk '{print $1}') \
      --seed=49 \
      --workers=${workers} \
      --learning-rate=${learning_rate} \
      --mom=0.9 \
      --weight-decay=1.0e-04  \
      --print-freq=1 \
      --dist-backend 'hccl' \
      --device='npu' \
      --gpu=${RANK_ID} \
      --epochs=${train_epochs} \
      --world-size=8 \
      --amp \
      --rank=${RANK_ID} \
      --batch-size=${batch_size} > ${test_path_dir}/output/${RANK_ID}/train_${RANK_ID}.log 2>&1 &
      
done

echo "------------------trian over------------------"

wait

##################获取训练数据##################
#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

# 结果打印，不需要修改
echo "------------------ Final result ------------------"
ASCEND_DEVICE_ID=0

#输出性能FPS，需要模型审视修改
FPS=`grep -a 'FPS'  ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk -F " " '{print $NF}'|awk 'END {print}'`
#打印，不需要修改
echo "Final Performance images/sec : $FPS"

#输出训练精度,需要模型审视修改
train_accuracy=`grep -a '* Acc@1' ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk 'END {print}'|awk -F "Acc@1" '{print $NF}'|awk -F " " '{print $1}'`
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
grep Epoch: ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|grep -v Test|awk -F "Loss" '{print $NF}' | awk -F " " '{print $1}' >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt

#最后一个迭代loss值，不需要修改
ActualLoss=`awk 'END {print}' ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt`

#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" > ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log

