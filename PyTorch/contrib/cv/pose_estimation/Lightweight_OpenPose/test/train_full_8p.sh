#!/bin/bash

export PYTHONWARNINGS="ignore:semaphore_tracker:UserWarning"
################基础配置参数，需要模型审视修改##################
# 必选字段(必须在此处定义的参数): Network batch_size RANK_SIZE
# 网络名称，同目录名称
Network="openpose"
# 训练batch_size
batch_size=640
# 训练使用的npu卡数
export RANK_SIZE=8
# 数据集路径,保持为空,不需要修改
data_path=""

# 训练epoch
train_epochs=280
# 加载数据进程数
workers=$(nproc)
# lr
base_lr=32e-5

# 参数校验，data_path为必传参数，其他参数的增删由模型自身决定；此处新增参数需在上面有定义并赋值
for para in $*
do
    if [[ $para == --device_id* ]];then
        device_id=`echo ${para#*=}`
    elif [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    elif [[ $para == --step* ]];then
        step=`echo ${para#*=}`
    elif [[ $para == --batch_size* ]];then
        batch_size=`echo ${para#*=}`
    fi
done

# 校验是否传入data_path,不需要修改
if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
    exit 1
fi

# 校验是否传入step,不需要修改
if [[ $step == "" ]];then
    echo "[Error] para \"step\" must be confing"
    exit 1
fi

# 根据step处理变量名
if [[ ${step} == "1" ]];then
  experiment_name="step_one"
  option="--from-mobilenet"
  model="mobilenet_sgd_68.848.pth.tar"
  refine_stages=1
elif [[ ${step} == "2" ]]; then
  experiment_name="step_two"
  option="--weights-only"
  model="model_best.pth"
  refine_stages=1
elif [[ ${step} == "3" ]]; then
  experiment_name="step_three"
  option="--weights-only"
  model="model_best.pth"
  refine_stages=3
fi

echo ${experiment_name}
echo ${option}
echo ${model}
echo ${refine_stages}

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

#################创建日志输出目录，不需要修改#################
ASCEND_DEVICE_ID=0
if [ -d ${test_path_dir}/output/${ASCEND_DEVICE_ID} ];then
    rm -rf ${test_path_dir}/output/${ASCEND_DEVICE_ID}
    mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID
else
    mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID
fi

#################启动训练脚本#################
#训练开始时间，不需要修改
start_time=$(date +%s)

# 非平台场景时source 环境变量
check_etp_flag=`env | grep etp_running_flag`
etp_flag=`echo ${check_etp_flag#*=}`
if [ x"${etp_flag}" != x"true" ];then
    source ${test_path_dir}/env_npu.sh
fi

RANK_ID_START=0
RANK_SIZE=8

for((RANK_ID=$RANK_ID_START;RANK_ID<$((RANK_SIZE+RANK_ID_START));RANK_ID++));
do

KERNEL_NUM=$(($(nproc)/8))
PID_START=$((KERNEL_NUM * RANK_ID))
PID_END=$((PID_START + KERNEL_NUM - 1))
taskset -c $PID_START-$PID_END python3 train.py \
    --train-images-folder ${data_path}/train2017/ \
    --prepared-train-labels ./prepared_train_annotation.pkl \
    --val-labels ./val_subset.json \
    --val-images-folder ${data_path}/val2017/ \
    --checkpoint-path ./${model} \
    ${option} \
    --num-refinement-stages ${refine_stages}\
    --base-lr=${base_lr} \
    --num-workers ${workers} \
    --epochs ${train_epochs} \
    --batch-size=${batch_size} \
    --experiment-name ${experiment_name} \
    --print-freq 1 \
    --addr=$(hostname -I |awk '{print $1}') \
    --rank=0 \
    --dist-url='tcp://127.0.0.1:50000' \
    --world-size=1 \
    --dist-backend 'hccl' \
    --amp \
    --gpu=${RANK_ID} \
    --loss-scale=16 \
    --opt-level O1 \
    --device-list '0,1,2,3,4,5,6,7' \
    --device="npu" \
      > ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &
done
wait

# save best model per step
if [ -f "model_best.pth" ];then
  cp model_best.pth "./${experiment_name}_checkpoints/"
else
  echo "model_best.pth don't exsit"
  exit 1
fi

#################获取训练数据################
#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
FPS=`grep -a 'FPS'  ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | awk -F " " '{print $11}'|awk 'END {print}'`
TrainAccuracy=$( grep "best model" ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | awk 'END {print}' | awk -F ":" '{print $2}')
#打印，不需要修改
echo "Final Performance images/sec : $FPS"
#打印，不需要修改
echo "E2E Training Duration sec : $e2e_time"
#打印，不需要修改
echo "TrainAccuracy : $TrainAccuracy"

#性能看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'acc'

#获取性能数据，不需要修改
#吞吐量
ActualFPS=${FPS}
#单迭代训练时长
TrainingTime=`awk 'BEGIN{printf "%.2f\n", '${batch_size}'*1000/'${FPS}'}'`

#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
# 根据step处理loss
if [[ ${step} == "3" ]];then
  grep Epoch: ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log  | \
     awk -F 'paf' '{print $2" "$3" "$4" "$5}' | \
     awk -F ' ' '{print $1","$4","$7","$10","$13","$16","$19","$22}' \
     > ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt
else
  grep Epoch: ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log  | \
     awk -F 'paf' '{print $2" "$3}' | \
     awk -F ' ' '{print $1","$4","$7","$10}' \
     > ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt
fi

#最后一个迭代loss值，不需要修改
ActualLoss=`awk 'END {print}'  ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt`

#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" >  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainAccuracy = ${TrainAccuracy}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log

# save train log
if [ -d "${test_path_dir}/output/${ASCEND_DEVICE_ID}/" ];then
  cp -rf "${test_path_dir}/output/${ASCEND_DEVICE_ID}/" "./${experiment_name}_checkpoints/"
else
  echo "log directory don't exsit"
  exit 1
fi
