#!/bin/bash

################基础配置参数，需要模型审视修改##################
# 必选字段(必须在此处定义的参数): Network batch_size RANK_SIZE
# 网络名称，同目录名称
Network="Conformer_Ti"
# 训练batch_size
batch_size=128
# 训练使用的npu卡数
export RANK_SIZE=1
# 数据集路径,保持为空,不需要修改
data_path=""

# 训练最大iter数
max_iter=10010


# 参数校验，data_path为必传参数，其他参数的增删由模型自身决定；此处新增参数需在上面有定义并赋值
for para in $*
do
    if [[ $para == --data_path* ]];then
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
echo ${pwd}

#################创建日志输出目录，不需要修改#################
ASCEND_DEVICE_ID=0
if [ -d ${test_path_dir}/output/${ASCEND_DEVICE_ID} ];then
    rm -rf ${test_path_dir}/output/${ASCEND_DEVICE_ID}
    mkdir -p ${test_path_dir}/output/${ASCEND_DEVICE_ID}
else
    mkdir -p ${test_path_dir}/output/${ASCEND_DEVICE_ID}
fi

# 变量
export SPACH_DATASETS=${data_path}
export PYTHONPATH=./:$PYTHONPATH

#################启动训练脚本#################
#训练开始时间，不需要修改
start_time=$(date +%s)
# 非平台场景时source 环境变量
check_etp_flag=`env | grep etp_running_flag`
etp_flag=`echo ${check_etp_flag#*=}`
if [ x"${etp_flag}" != x"true" ];then
    source ${test_path_dir}/env_npu.sh
    export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
fi

KERNEL_NUM=$(($(nproc)/8))
for i in $(seq 0 0)
do
    if [ $(uname -m) = "aarch64" ]
    then
        PID_START=$((KERNEL_NUM * i))
        PID_END=$((PID_START + KERNEL_NUM - 1))
        taskset -c $PID_START-$PID_END \
          python3 -u main.py \
            --model Conformer_tiny_patch16 \
            --data-set IMNET \
            --batch-size 128 \
            --lr 0.001 \
            --num_workers 4 \
            --epochs 2 \
            --data-path ${data_path} \
            --output_dir ${test_path_dir}/output/${ASCEND_DEVICE_ID} \
            > ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &
    else
        python3 -u main.py \
          --model Conformer_tiny_patch16 \
          --data-set IMNET \
          --batch-size 128 \
          --lr 0.001 \
          --num_workers 4 \
          --epochs 2 \
          --data-path ${data_path} \
          --output_dir ${test_path_dir}/output/${ASCEND_DEVICE_ID} \
        > ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &
    fi
done

wait


##################获取训练数据################
#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
FPS=`cat ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | grep 'Epoch' | grep 'FPS:'|awk -F " " '{print $4}'|awk 'END {print}'`
#打印，不需要修改
echo "Final Performance FPS : ${FPS}"

#打印，不需要修改
echo "E2E Training Duration sec : $e2e_time"

#性能看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'acc'

##获取性能数据，不需要修改
#吞吐量
AvgFPS=${FPS}

#最后一个迭代loss值
MinLoss=`cat ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | grep 'Averaged stats:' | awk 'BEGIN {min = 65536} {if ($12+0 < min+0) min=$12} END {print min}'`
MaxAccuracy=`cat ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | grep 'Max accuracy' | awk 'BEGIN {max = 0} {if ($3+0 > max+0) max=$3} END {print max}'`
echo "MaxAccuracy = ${MaxAccuracy}"
#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" >  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "AvgFPS = ${AvgFPS}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "MinLoss = ${MinLoss}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "MaxAccuracy = ${MaxAccuracy}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
