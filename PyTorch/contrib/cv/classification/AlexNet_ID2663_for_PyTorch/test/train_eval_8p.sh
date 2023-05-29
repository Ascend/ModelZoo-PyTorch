#!/bin/bash

################基础配置参数，需要模型审视修改##################
# 必选字段(必须在此处定义的参数): Network batch_size RANK_SIZE
# 网络名称，同目录名称
Network="AlexNet"
# 训练batch_size
batch_size=1024
# 训练使用的npu卡数
export RANK_SIZE=8
# ckpt文件路径
resume="./checkpoints/model_best.pth.tar"
# 数据集路径,修改为本地数据集路径
data_path=""

# 训练epoch
train_epochs=1
# 学习率
learning_rate=0.04

# 参数校验，data_path为必传参数，其他参数的增删由模型自身决定；此处新增参数需在上面有定义并赋值
for para in $*
do
    if [[ $para == --device_id* ]];then
        device_id=`echo ${para#*=}`
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
# 训练开始时间，不需要修改
start_time=$(date +%s)
# source 环境变量
# 非平台场景时source 环境变量
check_etp_flag=`env | grep etp_running_flag`
etp_flag=`echo ${check_etp_flag#*=}`
if [ x"${etp_flag}" != x"true" ];then
    source ${test_path_dir}/env_npu.sh
fi

python3 ./main.py \
	${data_path} \
    --evaluate \
    --resume ${resume} \
	-a alexnet \
	--addr=$(hostname -I |awk '{print $1}') \
	--seed=49 \
	--workers=$(nproc) \
	--learning-rate=${learning_rate} \
	--mom=0.9 \
	--weight-decay=1.0e-04  \
	--print-freq=1 \
    --dist-url='tcp://127.0.0.1:41111' \
    --dist-backend 'hccl' \
    --multiprocessing-distributed \
    --world-size=1 \
    --rank=0 \
    --device='npu' \
    --epochs=${train_epochs} \
    --amp \
    --label-smoothing=0.1 \
    --batch-size=${batch_size} > ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &
