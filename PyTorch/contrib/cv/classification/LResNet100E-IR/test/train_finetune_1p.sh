#!/bin/bash

source test/env_npu.sh;

################基础配置参数，需要模型审视修改##################
# 必选字段(必须在此处定义的参数): Network batch_size RANK_SIZE
# 网络名称，同目录名称
Network="LResNet100E"
# backbone mode [ir, ir_se]
net="ir_se"
# backbone depth [50,100,152]
depth=100
# 训练数据集
data_mode="emore"
# 验证数据集
eval_data_mode="lfw"
# 数据集路径,保持为空,不需要修改
data_path=""

# 训练最大iter数
max_iter=-1
# 从第几个epoch开始训练，默认为0
start_epoch=0
# 训练epoch
epoch=20
# 训练batch_size
batch_size=256
# 学习率
learning_rate=0.001
# 加载数据进程数
workers=8
# 随机种子
seed=2021

# 1为进行迁移学习 0为加载之前权重继续训练
is_finetune=1
# pretrained weights path
pth_path=""

# 训练使用的npu卡数
export RANK_SIZE=1
# 指定是gpu 还是npu 还是cpu
device_type='npu'
# 指定训练所使用的device卡id
device_id=0


# 是否进行分布式训练 
distributed=0

# 是否使用apex的半精度优化 0为false 1为true
use_amp=1
# "apex amp level, [O1, O2]"
opt_level="O2"
# "apex amp loss scale, [128.0, None]"
loss_scale=128.0


# 参数校验，data_path为必传参数，其他参数的增删由模型自身决定；此处新增参数需在上面有定义并赋值
for para in $*
do
    if [[ $para == --pth_path* ]];then
        pth_path=`echo ${para#*=}`
    elif [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    fi
done

# 校验是否传入data_path,不需要修改
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
if [ -d ${test_path_dir}/output/train_${RANK_SIZE} ];then
    rm -rf ${test_path_dir}/output/train_${RANK_SIZE}
    mkdir -p ${test_path_dir}/output/train_${RANK_SIZE}
else
    mkdir -p ${test_path_dir}/output/train_${RANK_SIZE}
fi

LOG_ROOT=${test_path_dir}/output/train_${RANK_SIZE}
# 变量
export DETECTRON2_DATASETS=${data_path}
export PYTHONPATH=./:$PYTHONPATH

#################启动训练脚本#################
# 训练开始时间，不需要修改
start_time=$(date +%s)

python3 train.py \
    --net_mode ${net} \
    --net_depth ${depth} \
    --data_mode ${data_mode} \
    --eval_data_mode ${eval_data_mode} \
    --data_path ${data_path} \
    --max_iter ${max_iter} \
    --start_epoch ${start_epoch} \
    --epochs ${epoch} \
    --batch_size ${batch_size} \
    --lr ${learning_rate} \
    --num_workers ${workers} \
    --seed ${seed} \
    --is_finetune ${is_finetune} \
    --resume ${pth_path} \
    --device_type ${device_type} \
    --device_id ${device_id} \
    --distributed ${distributed} \
    --use_amp ${use_amp} \
    --opt_level ${opt_level} \
    --loss_scale ${loss_scale} > ${LOG_ROOT}/train.log 2>&1

wait


##################获取训练数据################
# 性能看护结果汇总
# 训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'acc'

# 训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

# 结果打印，不需要修改
echo "------------------ Final result ------------------"
# 输出性能FPS，
cat ${LOG_ROOT}/train.log | grep "fps:" >> ${LOG_ROOT}/train_${CaseName}_fps.log
FPS=`cat ${LOG_ROOT}/train_${CaseName}_fps.log |grep 'fps:'| tail -n 1 | awk '{print $2}'`
# 打印，不需要修改
echo "Final Performance images/sec : $FPS"

# 输出训练精度
train_accuracy=`cat ${LOG_ROOT}/train.log | grep 'lfw_accuracy:' | tail -n 1 | awk '{print $2}'`

# 打印，不需要修改
echo "Final Train Accuracy : ${train_accuracy}"
# 输出时间
echo "E2E Training Duration sec : $e2e_time"

# 获取性能数据，不需要修改
# 吞吐量
ActualFPS=${FPS}
# 单迭代训练时长
TrainingTime=`awk 'BEGIN{printf "%.2f\n", '${batch_size}'*1000/'${FPS}'}'`

# 从train.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
cat ${LOG_ROOT}/train.log | grep "train_loss:" | awk '{print $5}' >> ${LOG_ROOT}/train_${CaseName}_loss.log

# 最后一个迭代loss值，不需要修改
ActualLoss=`awk 'END {print}' ${LOG_ROOT}/train_${CaseName}_loss.log`

# 关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" >  ${LOG_ROOT}/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >>  ${LOG_ROOT}/${CaseName}.log
echo "BatchSize = ${BatchSize}" >>  ${LOG_ROOT}/${CaseName}.log
echo "DeviceType = ${DeviceType}" >>  ${LOG_ROOT}/${CaseName}.log
echo "CaseName = ${CaseName}" >>  ${LOG_ROOT}/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >>  ${LOG_ROOT}/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >>  ${LOG_ROOT}/${CaseName}.log
echo "TrainAccuracy = ${train_accuracy}" >> ${LOG_ROOT}/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >>  ${LOG_ROOT}/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >>  ${LOG_ROOT}/${CaseName}.log
