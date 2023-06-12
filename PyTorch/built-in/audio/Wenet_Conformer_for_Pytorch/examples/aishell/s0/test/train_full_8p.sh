#!/bin/bash

source ./env_npu.sh

#当前路径,不需要修改
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
rank_size=8
export RANK_SIZE=${rank_size}
WORLD_SIZE=${rank_size}
# 数据集路径,保持为空,不需要修改
data_path=""

#网络名称,同目录名称,需要模型审视修改
Network="Wenet_Conformer_for_PyTorch"

#训练起始stage
stage=-1

#训练batch_size,,需要模型审视修改
batch_size=16

epochs=240


# 参数校验，data_path为必传参数，其他参数的增删由模型自身决定；此处新增参数需在上面有定义并赋值
for para in $*
do
    if [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    elif [[ $para == --stage* ]];then
        stage=$((`echo ${para#*=}`))
    elif [[ $para == --stop_stage* ]];then
        stop_stage=$((`echo ${para#*=}`))
    fi
done

#校验是否传入data_path,不需要修改
if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
    exit 1
fi

ASCEND_DEVICE_ID=0
#创建DeviceID输出目录，不需要修改
if [ -d $test_path_dir/output/${ASCEND_DEVICE_ID} ];then
    rm -rf $test_path_dir/output/$ASCEND_DEVICE_ID
    mkdir -p $test_path_dir/output/$ASCEND_DEVICE_ID
else
    mkdir -p $test_path_dir/output/$ASCEND_DEVICE_ID
fi


#################启动训练脚本#################

# 必要参数替换配置文件
cd $test_path_dir/..

start_time=$(date +%s)

nohup bash ../run.sh \
  --stage ${stage} \
  --GPUS 8 \
  --data $data_path \
  --test_epoch ${epochs} \
  --local_rank_id ${ASCEND_DEVICE_ID} \
  --test_output_dir $test_path_dir/output &

wait

##################获取训练数据################
# 训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(($end_time - $start_time))

# 训练用例信息，不需要修改
BatchSize=$((batch_size * rank_size))
DeviceType=$(uname -m)
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'acc'

# 结果打印，不需要修改
echo "------------------ Final result ------------------"
# 输出性能FPS，需要模型审视修改
grep "${epochs}th epoch e2e fps:" ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | awk -F "${epochs}th epoch e2e fps:" '{print $1, substr($2,0,6)}' &>${test_path_dir}/output/$ASCEND_DEVICE_ID/train_${CaseName}_fps.log
FPS=$(cat ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_${CaseName}_fps.log | xargs echo -n | tr ' ' '+' | xargs echo | bc)
# 打印，不需要修改
echo "Final Performance iters/sec : $FPS"

# 输出训练精度,需要模型审视修改
train_loss=$(grep -a "Epoch $((epochs-1)) CV info cv_loss" ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | awk "Epoch $((epochs-1)) CV info cv_loss" '{print $2}' | awk 'END {print}')
# 打印，不需要修改
echo "Final Train Accuracy : ${train_loss}"
echo "E2E Training Duration sec : $e2e_time"

# 性能看护结果汇总
# 获取性能数据，不需要修改
# 吞吐量
ActualFPS=${FPS}
# 单迭代训练时长
TrainingTime=$(awk 'BEGIN{printf "%.2f\n", '${BatchSize}'*1000/'${FPS}'}')

# 从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
grep "CV info cv_loss" ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | awk -F "CV info cv_loss" '{print $2}' >>${test_path_dir}/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt

# 最后一个迭代loss值，不需要修改
ActualLoss=$(awk 'END {print}' ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt)

# 关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" >${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${WORLD_SIZE}" >>${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >>${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >>${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >>${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >>${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >>${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainLoss = ${train_loss}" >>${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >>${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
