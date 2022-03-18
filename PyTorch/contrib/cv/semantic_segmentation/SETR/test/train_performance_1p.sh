#!/usr/bin/env bash

################基础配置参数，需要模型审视修改##################
# 必选字段(必须在此处定义的参数): Network batch_size RANK_SIZE
# 网络名称，同目录名称
Network="SETR"

# 设置训练所用参数
CONFIG='configs/SETR/SETR_Naive_768x768_40k_cityscapes_npu.py'


# 如果是训练重启，请选择大于0的数，并将地址填入下表
is_resume=0             
# resume_work_dirs=work_dirs/training_npu_1p_job_20210927224806
# 选择运行的GPU id
device_id=0
# 训练使用的npu卡数
GPUS=1
#训练batch_size,,需要模型审视修改
batch_size=1
# 设置通讯端口
PORT=${PORT:-29699}     
# 设置数据集地址
data_path='data/cityscapes/'

#校验是否传入data_path,不需要修改
for para in $*
do
    if [[ $para == --device_id* ]];then
        device_id=`echo ${para#*=}`
    elif [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    fi
done

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

#训练开始时间，不需要修改
start_time=$(date +%s)
echo "start_time: ${start_time}"

currentDir=$(cd "$(dirname "$0")";cd ..;pwd)
currtime=`date +%Y%m%d%H%M%S`
train_log_dir=${currentDir}/work_dirs/training_performance_npu_1p_job_${currtime}

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
if [ -d ${test_path_dir}/output/${ASCEND_DEVICE_ID} ];then
    rm -rf ${test_path_dir}/output/${ASCEND_DEVICE_ID}
    mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID
else
    mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID
fi


#非平台场景时source 环境变量
check_etp_flag=`env | grep etp_running_flag`
etp_flag=`echo ${check_etp_flag#*=}`
if [ x"${etp_flag}" != x"true" ];then
    source ${currentDir}/env_npu.sh
fi

export COMBINED_ENABLE=1

if [ ! -d "./work_dirs" ]; then
  mkdir ./work_dirs
fi

if [ $is_resume -gt 0 ]; then
        echo "正在拉起训练"
        train_log_dir=${currentDir}/${resume_work_dirs}
        echo "train log path is ${train_log_dir}"

        cd ${currentDir}
        python3  -m torch.distributed.launch \
        --nproc_per_node=$GPUS \
        --master_port=$PORT \
        tools/train.py \
        ${CONFIG} \
        --launcher pytorch \
        --use_npu True \
        --gpu_ids ${device_id} \
        --work-dir ${train_log_dir} \
        --options data_root=${data_path} data.train.data_root=${data_path} data.val.data_root=${data_path} data.test.data_root=${data_path} \
        --use_amp True \
        --sys_fp_16 False \
        --apex_opt_level O1 \
        --loss_scale_value 128 \
        --seed 19 \
        --resume-from ${resume_work_dirs}/latest.pth \
        > ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &
else
        mkdir -p ${train_log_dir}
        echo "train log path is ${train_log_dir}"
        cd ${currentDir}
        python3  -m torch.distributed.launch \
        --nproc_per_node=$GPUS \
        --master_port=$PORT \
        tools/train.py \
        ${CONFIG} \
        --launcher pytorch \
        --use_npu True \
        --gpu_ids ${device_id} \
        --work-dir ${train_log_dir} \
        --options data_root=${data_path} data.train.data_root=${data_path} data.val.data_root=${data_path} data.test.data_root=${data_path} \
        --use_amp False \
        --sys_fp_16 True \
        --apex_opt_level O1 \
        --loss_scale_value 128 \
        --total_iters 200 \
        --seed 19 \
        > ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &
fi


wait

##################获取训练数据################
#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
FPS=`grep -a ' train_FPS: '  ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk -F " " '{print $9}'|awk 'END {print}'`
#打印，不需要修改
echo "Final Performance images/sec : $FPS"

#打印，不需要修改
echo "E2E Training Duration sec : $e2e_time"

#性能看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${GPUS}'p'_'acc'

##获取性能数据，不需要修改
#吞吐量
ActualFPS=${FPS}
#单迭代训练时长
TrainingTime=`awk 'BEGIN{printf "%.2f\n", '${batch_size}'*1000/'${FPS}'}'`

#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
grep 'Iter ' ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|grep -v Test|awk -F "loss: " '{print $NF}' | awk -F " " '{print $1}' >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt

#最后一个迭代loss值，不需要修改
ActualLoss=`awk 'END {print}'  ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt`

#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" >  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${GPUS}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainAccuracy = ${train_accuracy}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log