#!/bin/bash

export PYTHONWARNINGS="ignore:semaphore_tracker:UserWarning"
################基础配置参数，需要模型审视修改##################
# 必选字段(必须在此处定义的参数): Network batch_size RANK_SIZE
# 网络名称，同目录名称
Network="openpose"
# 训练batch_size
batch_size=80
# 训练使用的npu卡数
export RANK_SIZE=1
# 数据集路径,保持为空,不需要修改
data_path=""

# 训练epoch
train_epochs=280
# 加载数据进程数
workers=16
# lr
base_lr=4e-5

# 参数校验，data_path为必传参数，其他参数的增删由模型自身决定；此处新增参数需在上面有定义并赋值
for para in $*
do
    if [[ $para == --device_id* ]];then
        device_id=`echo ${para#*=}`
    elif [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    elif [[ $para == --step* ]];then
        step=`echo ${para#*=}`
    elif [[ $para == --checkpoint_path* ]];then
        checkpoint_path=`echo ${para#*=}`
    fi
done

# 校验是否传入data_path,不需要修改
if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
    exit 1
fi

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
ASCEND_DEVICE_ID=${device_id}
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

python3 train.py \
    --train-images-folder ${data_path}/train2017/ \
    --prepared-train-labels ./prepared_train_annotation.pkl \
    --val-labels ${data_path}/annotations/person_keypoints_val2017.json \
    --val-images-folder ${data_path}/val2017/ \
    --checkpoint-path ${checkpoint_path} \
    --weights-only \
    --num-workers ${workers} \
    --epochs ${train_epochs} \
    --batch-size=${batch_size} \
    --gpu=0 \
    --world-size=1 \
    --loss-scale=16 \
    --opt-level O1 \
    --device-list ${device_id} \
    --evaluate \
    --device="npu" \
     > ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &

wait

#################获取训练数据################
#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
# FPS=`grep -a 'FPS'  ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | awk -F " " '{print $11}'|awk 'END {print}'`
EvalAccuracy=$( grep "final eval acc" ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | awk -F ":" '{print $2}')
#打印，不需要修改
# echo "Final Performance images/sec : $FPS"
#打印，不需要修改
echo "E2E Training Duration sec : $e2e_time"
#打印，不需要修改
echo "EvalAccuracy : $EvalAccuracy"

#性能看护结果汇总
#训练用例信息，不需要修改
DeviceType=`uname -m`
CaseName=${Network}_${RANK_SIZE}'p'_'eval'

##获取性能数据，不需要修改
##吞吐量
#ActualFPS=${FPS}
##单迭代训练时长
#TrainingTime=`awk 'BEGIN{printf "%.2f\n", '${batch_size}'*1000/'${FPS}'}'`

#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" >  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
#echo "ActualFPS = ${ActualFPS}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
#echo "TrainingTime = ${TrainingTime}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "EvalAccuracy = ${EvalAccuracy}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log

rm -rf "eval_step${step}"
mkdir "eval_step${step}"
# save eval log
if [ -d "${test_path_dir}/output/${ASCEND_DEVICE_ID}/" ];then
  cp -rf "${test_path_dir}/output/${ASCEND_DEVICE_ID}/" "eval_step${step}"
else
  echo "log directory don't exsit"
  exit 1
fi