#!/bin/bash

# 网络名称
Network="BiLstm-CRF_RT1_ID4122_for_PyTorch"
batch_size=64
train_epochs=1
print_step=1
RANK_SIZE=1

# 数据集路径，保持为空，不需要修改
data_path=""

# 使能runtime1.0
export ENABLE_RUNTIME_V2=0
# 二进制
bin_mode=true

# NPU使用id
ASCEND_DEVICE_ID=${ASCEND_DEVICE_ID:-0}

# 参数校验，不需要修改
for para in $*; do
    if [[ $para == --more_path1* ]]; then
        more_path1=`echo ${para#*=}`
    elif [[ $para == --data_path* ]]; then
        data_path=`echo ${para#*=}`
    elif [[ $para == --batch_size* ]]; then
        batch_size=`echo ${para#*=}`
    elif [[ $para == --bin_mode* ]]; then
        bin_mode=`echo ${para#*=}`
    elif [[ $para == --amp_opt_level* ]]; then
        amp_opt_level=`echo ${para#*=}`
    elif [[ $para == --train_epochs* ]]; then
        train_epochs=`echo ${para#*=}`
    elif [[ $para == --print_step* ]]; then
        print_step=`echo ${para#*=}`
    elif [[ $para == --profiling* ]]; then
        profiling=`echo ${para#*=}`
    elif [[ $para == --p_start_step* ]]; then
        p_start_step=`echo ${para#*=}`
    elif [[ $para == --iteration_num* ]]; then
        iteration_num=`echo ${para#*=}`
    fi
done

# 校验是否传入data_path, 不需要修改
if [[ $data_path == "" ]]; then
    echo "[Error] param \"data_path\" must be config"
    exit 1
fi

if [[ "${profiling}" == "GE" ]]; then
    export GE_PROFILING_TO_STD_OUT=1
fi

###############指定训练脚本路径###############
# cd 到与test文件夹同层级目录下执行脚本，提高兼容性；test_path_dir为包含test文件夹的路径
cur_path=$(pwd)
cur_path_last_dirname=${cur_path##*/}
if [ x"${cur_path_last_dirname}" == x"test" ]; then
    test_path_dir=${cur_path}
    cd ..
    cur_path=$(pwd)
else
    test_path_dir=${cur_path}/test
fi

export PYTHONPATH=$PYTHONPATH:${cur_path}/named_entity_recognition

###############创建日志输出目录###############
if [ -d ${test_path_dir}/output/${ASCEND_DEVICE_ID} ]; then
    rm -rf ${test_path_dir}/output/${ASCEND_DEVICE_ID}
fi
mkdir -p ${test_path_dir}/output/${ASCEND_DEVICE_ID}

if [ -d ${cur_path}/BiLstm/ckpts ]; then
    rm -rf ${cur_path}/BiLstm/ckpts
fi
mkdir -p ${cur_path}/BiLstm/ckpts

# 非平台场景时source 环境变量
# check_etp_flag=$(env | grep etp_running_flag)
# etp_flag=$(echo ${check_etp_flag#*=})
# if [ x"${etp_flag}" != x"true" ];then
#     source ${test_path_dir}/env_npu.sh
# fi


# 训练开始时间
start_time=$(date +%s)

cd ${cur_path}
# training
nohup python3 -u runner.py \
    --data-path=${data_path} \
    --amp_opt_level=${amp_opt_level:-"O2"} \
    --local_rank=${ASCEND_DEVICE_ID} \
    --bin_mode=${bin_mode} \
    --train_epochs=${train_epochs} \
    --batch_size=${batch_size} \
    --print_step=${print_step} \
    --profiling=${profiling:-"false"} \
    --p_start_step=${p_start_step:-0} \
    --iteration_num=${iteration_num:-"-1"} \
    > ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &

wait

# 训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(($end_time - $start_time))

# 结果打印，不需要需改
echo "---------------- Final result ----------------"
sed -i "s|\r|\n|g" ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log

# 输出性能FPS
average_step_time=$(grep "step_time" ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | awk -F ":" '{print $4}' | awk 'BEGIN{count=0}{if(NR>3){sum+=$NF;count+=1}}END{printf "%.4f\n", sum/count}')
FPS=$(awk 'BEGIN{printf "%.2f\n", '${batch_size}'/'${average_step_time}'}')

# 输出训练精度
train_accuracy=$(grep "step/total_step" ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | awk -F "Loss:" '{print $2}' | awk 'END{print $1}')

# 最后一次迭代的Loss值
ActualLoss=$(grep "step/total_step" ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | awk -F "Loss:" '{print $2}' | awk 'END{print $1}')

# 打印，不需要修改
echo "Final performance images/sec: ${FPS}"
echo "Final train accuracy: ${train_accuracy}"
echo "ActualLoss: ${ActualLoss}"
echo "E2E training duration sec: ${e2e_time}"

###############看护结果汇总###############
DeviceType=$(uname -m)
CaseName=${Network}_bs${batch_size}_${RANK_SIZE}'p'_'perf'
# 单迭代训练时长
TrainingTime=$(awk 'BEGIN{printf "%.2f\n", '${batch_size}'*1000/'${FPS}'}')

# 输出所有的loss
grep "step/total_step" ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | awk -F "Loss:" '{print $2}' | awk '{print $1}' >> ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${CaseName}_loss.txt

# 关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" > ${test_path_dir}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >> ${test_path_dir}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "BatchSize = ${batch_size}" >> ${test_path_dir}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> ${test_path_dir}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "CaseName = ${CaseName}" >> ${test_path_dir}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "ActualFPS = ${FPS}" >> ${test_path_dir}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >> ${test_path_dir}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "TrainAccuracy = ${train_accuracy}" >> ${test_path_dir}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> ${test_path_dir}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> ${test_path_dir}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
# echo "Status = 0" >> ${test_path_dir}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
