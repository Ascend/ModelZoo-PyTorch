#!/bin/bash

################基础配置参数，需要模型审视修改##################
# 必选字段(必须在此处定义的参数): Network batch_size RANK_SIZE
# 网络名称，同目录名称
Network="EDSR_x2"
# 训练batch_size
batch_size=16
# 训练使用的npu卡数
export RANK_SIZE=1
# 数据集路径,保持为空,不需要修改
data_path=""
pre_train_model=""

# 指定训练所使用的npu device卡id
device_id=0
# 加载数据进程数
workers=128

# 参数校验，data_path为必传参数，其他参数的增删由模型自身决定；此处新增参数需在上面有定义并赋值
for para in $*
do
    if [[ $para == --device_id* ]];then
        device_id=`echo ${para#*=}`
    elif [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    elif [[ $para == --pre_train_model* ]];then
        pre_train_model=`echo ${para#*=}`
    fi
done

# 校验是否传入data_path,不需要修改
if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
    exit 1
fi
if [[ $pre_train_model == "" ]];then
    echo "[Error] para \"pre_train_model\" must be confing"
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

output_dir=${test_path_dir}/output/eval_8p
#################创建日志输出目录，不需要修改#################
if [ -d ${output_dir} ];then
    rm -rf ${output_dir}
    mkdir -p ${output_dir}
else
    mkdir -p ${output_dir}
fi
output_log=${output_dir}/test_eval_8p.log

#################启动训练脚本#################
#训练开始时间，不需要修改
start_time=$(date +%s)
# 非平台场景时source 环境变量
check_etp_flag=`env | grep etp_running_flag`
etp_flag=`echo ${check_etp_flag#*=}`
if [ x"${etp_flag}" != x"true" ];then
    source ${test_path_dir}/env_npu.sh
fi
python3 main_8p.py \
    --dir_data ${data_path} \
    --save  ${output_dir}/npu_8p/test \
    --seed 49 \
    --amp \
    --device npu \
    --checkpoint_path ${pre_train_model} \
    --dist-backend "hccl" \
    --test_only --data_range "801-900" > ${output_log} 2>&1 &

wait

##################获取训练数据################
#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))    


#结果打印，不需要修改
echo "------------------ Final result ------------------"

# 输出训练精度,需要模型审视修改
test_accuracy=`grep -a 'PSNR:' ${output_log} | tail -1 | awk -F " " '{print $2}'`
#打印，不需要修改
echo "Final Train Accuracy (PSNR): ${test_accuracy}"
echo "E2E Training Duration sec : $e2e_time"

# 训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'acc'


# 最后一个迭代loss值，不需要修改
ActualLoss=`grep Test ${output_log} | awk '{print $8}' | awk 'END {print}'`

#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" >  ${output_dir}/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >>  ${output_dir}/${CaseName}.log
echo "BatchSize = ${BatchSize}" >>  ${output_dir}/${CaseName}.log
echo "DeviceType = ${DeviceType}" >>  ${output_dir}/${CaseName}.log
echo "CaseName = ${CaseName}" >>  ${output_dir}/${CaseName}.log
echo "TrainAccuracy = ${test_accuracy}" >> ${output_dir}/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >>  ${output_dir}/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >>  ${output_dir}/${CaseName}.log