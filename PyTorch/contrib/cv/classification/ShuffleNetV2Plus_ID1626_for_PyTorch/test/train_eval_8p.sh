#!/bin/bash

################基础配置参数，需要模型审视修改##################
# 必选字段(必须在此处定义的参数): Network batch_size resume RANK_SIZE
# 网络名称，同目录名称
Network="ShuffleNetV2Plus_ID1626_for_PyTorch"
# 训练batch_size
batch_size=8192
# 训练使用的npu卡数
export RANK_SIZE=8
# ckpt文件路径,按实际路径配置
resume=/home/checkpoint.pth.tar
# 数据集路径,保持为空,不需要修改
data_path=""

# 训练epoch, 默认值360
train_epochs=360
# 指定8卡训练所使用的device_id_list
device_id_list=0,1,2,3,4,5,6,7
# 学习率
learning_rate=1

# 参数校验，data_path为必传参数，其他参数的增删由模型自身决定；此处新增参数需在上面有定义并赋值
for para in $*
do
    if [[ $para == --workers* ]];then
        workers=`echo ${para#*=}`
    elif [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
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
cur_path_last_diename=${cur_path##*/}
if [ x"${cur_path_last_diename}" == x"test" ];then
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
python3 ./train.py \
    --evaluate \
    --resume ${resume} \
    --data=${data_path} \
    --addr=$(hostname -I |awk '{print $1}') \
    --workers=128 \
    --print-freq=1 \
    --eval-freq=5 \
    --dist-url='tcp://127.0.0.1:50000' \
    --dist-backend='hccl' \
    --multiprocessing-distributed \
    --world-size=1 \
    --rank=0 \
    --amp \
    --batch-size ${batch_size} \
    --epochs=${train_epochs} \
    --learning-rate ${learning_rate} \
    --wd 8e-5 \
    --device-list=${device_id_list} \
    --device_num 8 \
    --benchmark 0 > ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &

wait

##################获取训练数据################
# 训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

# 结果打印，不需要修改
echo "------------------ Final result ------------------"
# 输出训练精度,需要模型审视修改
train_accuracy=`grep -a '* Acc@1'  ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk 'END {print}'|awk -F "Acc@1" '{print $NF}'|awk -F " " '{print $1}'`
# 打印，不需要修改
echo "Final Train Accuracy : ${train_accuracy}"
echo "E2E Training Duration sec : $e2e_time"

# 训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'eval'

# 最后一个迭代loss值，不需要修改
ActualLoss=`grep Test ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_${ASCEND_DEVICE_ID}.log | awk '{print $8}' | awk 'END {print}'`

# 关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" >  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainAccuracy = ${train_accuracy}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log