#!/bin/bash

################基础配置参数，需要模型审视修改##################
# 必选字段(必须在此处定义的参数): Network batch_size RANK_SIZE
# 网络名称，同目录名称
Network="ArcFace_ID4078_for_PyTorch"


# 训练使用的npu卡数
export RANK_SIZE=1
RANK_ID_START=0
# 模型结构
arch="arcface"
#训练epoch数
train_epochs=1
# 数据集路径,保持为空,不需要修改
data_path=""
precision_mode="allow_mix_precision"
#设置训练步数
perf_steps=200

#使能profiling，默认为False
profiling=False
start_step=90
stop_step=100

# 参数校验，data_path为必传参数，其他参数的增删由模型自身决定；此处新增参数需在上面有定义并赋值
for para in $*
do
    if [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    elif [[ $para == --precision_mode* ]];then
        precision_mode=`echo ${para#*=}`
    elif [[ $para == --conda_name* ]];then
        conda_name=`echo ${para#*=}`
        source set_conda.sh
        echo "conda_name: $conda_name"
        source activate $conda_name
    elif [[ $para == --profiling* ]];then
        profiling=`echo ${para#*=}`
    elif [[ $para == --start_step* ]];then
        start_step=`echo ${para#*=}`
    elif [[ $para == --stop_step* ]];then
        stop_step=`echo ${para#*=}`
    fi
done

if [[ $profiling == "GE" ]];then
    export GE_PROFILING_TO_STD_OUT=1
    profiling=True
elif [[ $profiling == "CANN" ]];then
    profiling=True
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

#export NODE_RANK=0
#################启动训练脚本#################
# 训练开始时间，不需要修改
start_time=$(date +%s)
# 非平台场景时source 环境变量
check_etp_flag=`env | grep etp_running_flag`
etp_flag=`echo ${check_etp_flag#*=}`
if [ x"${etp_flag}" != x"true" ];then
    source ${test_path_dir}/env_npu.sh
    perf_steps=2000
fi

sed -i "s|`grep 'config.rec' ${cur_path}/configs/glint360k_r100.py|awk -F " " '{print $3}'`|'"$data_path"'|g" ${cur_path}/configs/glint360k_r100.py
sed -i "s|config.num_epoch = 20|config.num_epoch = $train_epochs|g" ${cur_path}/configs/glint360k_r100.py

if [[ $precision_mode == "must_keep_origin_dtype" ]];then
    sed -i "s|config.fp16 = True|config.fp16 = False|g" ${cur_path}/configs/glint360k_r100.py
fi
for((RANK_ID=$RANK_ID_START;RANK_ID<$((RANK_SIZE+RANK_ID_START));RANK_ID++));
do
    # 设置环境变量，不需要修改
    export RANK=$RANK_ID
    ASCEND_DEVICE_ID=$RANK_ID

    # 创建DeviceID输出目录，不需要修改
    if [ -d ${test_path_dir}/output/${ASCEND_DEVICE_ID} ];then
        rm -rf ${test_path_dir}/output/${ASCEND_DEVICE_ID}
        mkdir -p ${test_path_dir}/output/${ASCEND_DEVICE_ID}
    else
        mkdir -p ${test_path_dir}/output/${ASCEND_DEVICE_ID}
    fi
    
    KERNEL_NUM=$(($(nproc)/8))
    PID_START=$((KERNEL_NUM * RANK_ID))
    PID_END=$((PID_START + KERNEL_NUM - 1))

    taskset -c $PID_START-$PID_END python3 -u train.py \
        configs/glint360k_r100.py \
        --profiling ${profiling} \
        --start_step ${start_step} \
        --stop_step ${stop_step} \
        --local_rank=${RANK_ID} --perf_steps=$perf_steps > ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &
done
wait

##################获取训练数据################

# 训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

sed -i "s|config.num_epoch = $train_epochs|config.num_epoch = 20|g" ${cur_path}/configs/glint360k_r100.py

training_log=${test_path_dir}/output/${ASCEND_DEVICE_ID}/training_${ASCEND_DEVICE_ID}.log
grep "Training" ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log > ${training_log}

# 训练用例信息，不需要修改
BatchSize=`grep "total_batch_size" ${training_log} |awk '{print $5}'`
DeviceType=`uname -m`
if [[ $precision_mode == "must_keep_origin_dtype" ]];then
        CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'fp32'_'perf'
else
        CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'perf'
fi


# 结果打印，不需要修改
echo "------------------ Final result ------------------"
# 输出性能FPS，需要模型审视修改
grep "Speed" ${training_log} |awk '{print $4}' > ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${CaseName}_fps.log
FPS=`cat ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${CaseName}_fps.log |tail -n 100 |awk '{a+=$1} END {if (NR != 0) printf("%.3f",a/NR)}'`
# 打印，不需要修改
echo "Final Performance images/sec : $FPS"
#输出编译时间
CompileTime=`grep step_time ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log| head -2 |awk -F "step_time = " '{print $2}' | awk '{sum+=$1} END {print"",sum}' |sed s/[[:space:]]//g`

# 输出训练精度,需要模型审视修改
#lfw_accuracy_log=${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${CaseName}_lfw_accuracy.log
#cfp_fp_accuracy_log=${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${CaseName}_cfp_fp_accuracy.log
#agedb_30_accuracy_log=${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${CaseName}_agedb_30_accuracy.log

#grep "lfw" ${training_log} > ${lfw_accuracy_log}
#grep "cfp_fp" ${training_log} > ${cfp_fp_accuracy_log}
#grep "agedb_30" ${training_log} > ${agedb_30_accuracy_log}

#train_lfw_accuracy_highest=`grep -a 'Accuracy-Highest' ${lfw_accuracy_log} |awk 'END {print $4}'`
#train_cfp_fp_accuracy_highest=`grep -a 'Accuracy-Highest' ${cfp_fp_accuracy_log} |awk 'END {print $4}'`
#train_agedb_30_accuracy_highest=`grep -a 'Accuracy-Highest' ${agedb_30_accuracy_log} |awk 'END {print $4}'`
#train_accuracy="'lfw': ${train_lfw_accuracy_highest} 'cfp_fp': ${train_cfp_fp_accuracy_highest} 'agedb_30': ${train_agedb_30_accuracy_highest}"

# 打印，不需要修改
#echo "Final Train Accuracy : ${train_accuracy}"
echo "E2E Training Duration sec : $e2e_time"

# 性能看护结果汇总
# 获取性能数据，不需要修改
# 吞吐量
ActualFPS=${FPS}
# 单迭代训练时长
TrainingTime=`awk 'BEGIN{printf "%.2f\n", '${BatchSize}'*1000/'${FPS}'}'`

# 从training_log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
grep -rns "Loss" ${training_log} |awk -F " " '{print $7}' > ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${CaseName}_loss.txt

# 倒数第二个迭代loss值，不需要修改
ActualLoss=`tail -n 2 ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${CaseName}_loss.txt |awk 'NR==1 {print}'`

# 关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" >  ${test_path_dir}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >>  ${test_path_dir}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "BatchSize = ${BatchSize}" >>  ${test_path_dir}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "DeviceType = ${DeviceType}" >>  ${test_path_dir}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "CaseName = ${CaseName}" >>  ${test_path_dir}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >>  ${test_path_dir}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >>  ${test_path_dir}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
#echo "TrainAccuracy = ${train_accuracy}" >> ${test_path_dir}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >>  ${test_path_dir}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >>  ${test_path_dir}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "CompileTime = ${CompileTime}" >> ${test_path_dir}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
