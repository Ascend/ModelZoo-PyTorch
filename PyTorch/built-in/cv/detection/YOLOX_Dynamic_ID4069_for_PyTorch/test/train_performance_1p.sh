#!/bin/bash
cur_time=`date +%Y%m%d%H%M%S`

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

#conda环境的名称
conda_name=py4

# 数据集路径,保持为空,不需要修改
data_path=""

#网络名称,同目录名称,需要模型审视修改
Network="YOLOX_Dynamic_ID4069_for_PyTorch"

#训练batch_size,,需要模型审视修改
batch_size=16

#训练epoch，不需要修改
epochs=1

# 指定训练所使用的npu device卡id
device_id=

precision_mode="allow_mix_precision"

# 参数校验，data_path为必传参数，其他参数的增删由模型自身决定；此处新增参数需在上面有定义并赋值
for para in $*
do
    if [[ $para == --precision_mode* ]];then
        precision_mode=`echo ${para#*=}`
    elif [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    elif [[ $para == --device_id* ]];then
        device_id=`echo ${para#*=}`
    elif [[ $para == --batch_size* ]];then
        batch_size=`echo ${para#*=}`
    elif [[ $para == --epochs* ]];then
        epochs=`echo ${para#*=}`
    elif [[ $para == --conda_name* ]];then
        conda_name=`echo ${para#*=}`
        source $test_path_dir/set_conda.sh
        source activate $conda_name
    fi
done

#校验是否传入data_path,不需要修改
if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
    exit 1
fi

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

if [ ${precision_mode} == "must_keep_origin_dtype" ];then
    adv_param=""
else
    adv_param=" --fp16 "
fi

if [ -d ${test_path_dir}/output/${ASCEND_DEVICE_ID} ];then
    rm -rf ${test_path_dir}/output/${ASCEND_DEVICE_ID}
    mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID
else
    mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID
fi
# 指定单卡训练卡id
export NPU_CALCULATE_DEVICE=$ASCEND_DEVICE_ID

# 指定数据集路径
export YOLOX_DATADIR=$data_path

echo "Current CMD: $0 $1 $2 $3"
# runtime2.0开关
export ENABLE_RUNTIME_V2=1
echo "Runtime2.0 : $ENABLE_RUNTIME_V2"
echo "Runtime2.0 BLACKLIST: $RUNTIME_V2_BLACKLIST"

sed -i "s|if self.iter >= 100:pass|if self.iter >= 1000:break|g" ${test_path_dir}/../yolox/core/trainer.py
sed -i "s|for self.epoch in range(self.start_epoch, self.max_epoch):|for self.epoch in range(self.start_epoch, 1):|g" ${test_path_dir}/../yolox/core/trainer.py

#################启动训练脚本#################

# 非平台场景时source 环境变量
check_etp_flag=`env | grep etp_running_flag`
etp_flag=`echo ${check_etp_flag#*=}`
if [ x"${etp_flag}" != x"true" ];then
    source ${test_path_dir}/env_npu.sh
fi

start_time=$(date +%s)
KERNEL_NUM=$(($(nproc)/8))
PID_START=$((KERNEL_NUM * ASCEND_DEVICE_ID))
PID_END=$((PID_START + KERNEL_NUM - 1))
taskset -c $PID_START-$PID_END python3 -m yolox.tools.train -n yolox-s -d 1 -b ${batch_size} ${adv_param} -f exps/example/yolox_voc/yolox_voc_s.py > ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &
wait

##################获取训练数据################
#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

sed -i "s|if self.iter >= 10:break|if self.iter >= 10:pass|g" ${test_path_dir}/../yolox/core/trainer.py
sed -i "s|for self.epoch in range(self.start_epoch, 1):|for self.epoch in range(self.start_epoch, self.max_epoch):|g" ${test_path_dir}/../yolox/core/trainer.py

iter_time=`grep 'iter_time:' ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log |awk -F 'iter_time:' '{print $2}'|awk -F 's' '{print $1}'|tail -n+3|awk '{sum+=$1} END {print"",sum/NR}'|sed s/[[:space:]]//g`
FPS=`awk 'BEGIN{printf "%.2f\n", '${batch_size}'/'${iter_time}'}'`

#输出CompileTime
CompileTime=`grep 'iter_time:' ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log |head -n 2|awk -F 'iter_time:' '{print $2}'|awk -F 's' '{print $1}'|awk '{sum+=$1} END {print sum}'|sed s/[[:space:]]//g`

# 输出训练精度
train_accuracy=`grep 'Training of experiment is done and the best AP' ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log |awk -F 'Training of experiment is done and the best AP is' '{print $2}'|awk 'NR==1{max=$1;next}{max=max>$1?max:$1}END{print max}'|sed s/[[:space:]]//g`
echo "Final Performance images/sec : $FPS"
echo "E2E Training Duration sec : $e2e_time"

#性能看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`

if [[ $precision_mode == "must_keep_origin_dtype" ]];then
    CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'fp32'_'perf'
else
    CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'perf'
fi
#获取性能数据，不需要修改
#吞吐量
ActualFPS=${FPS}
#单迭代训练时长
TrainingTime=`awk 'BEGIN{printf "%.2f\n", '${batch_size}'*1000/'${FPS}'}'`

#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要模型审视修改
grep  "total_loss:" $test_path_dir/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log |awk -F 'total_loss: ' '{print $2}'|awk -F ',' '{print $1}'  > ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt
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
echo "TrainAccuracy = ${train_accuracy}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CompileTime = ${CompileTime}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log