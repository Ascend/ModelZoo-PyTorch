#!/bin/bash

################基础配置参数，需要模型审视修改##################
# 必选字段(必须在此处定义的参数): Network batch_size RANK_SIZE
# 网络名称，同目录名称
Network="MAE"
# 训练batch_size
batch_size=128
# 训练使用的npu卡数
export RANK_SIZE=8
export RANK_ID_START=0
# 数据集路径,保持为空,不需要修改
data_path=""
# 预训练模型权重文件路径
finetune_pth=""
# 模型保存目录
output_dir="output_finetune_full__large_8p"

# 训练epoch
train_epochs=1
# 加载数据进程数
workers=32

# 参数校验，data_path为必传参数，其他参数的增删由模型自身决定；此处新增参数需在上面有定义并赋值
for para in $*
do
    if [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    elif [[ $para == --finetune_pth* ]];then
        finetune_pth=`echo ${para#*=}`
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

export NODE_RANK=0
#################启动训练脚本#################
# 训练开始时间，不需要修改
start_time=$(date +%s)
# 非平台场景时source 环境变量
check_etp_flag=`env | grep etp_running_flag`
etp_flag=`echo ${check_etp_flag#*=}`
if [ x"${etp_flag}" != x"true" ];then
    source ${test_path_dir}/env_npu.sh
fi

export WORLD_SIZE=8
KERNEL_NUM=$(($(nproc)/8))
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=23333

for((RANK_ID=0;RANK_ID<RANK_SIZE;RANK_ID++))
do
export RANK=$RANK_ID
export LOCAL_RANK=$RANK_ID
PID_START=$((KERNEL_NUM * RANK_ID))
PID_END=$((PID_START + KERNEL_NUM -1))
nohup taskset -c $PID_START-$PID_END python3 -u  main_finetune.py \
             --local_rank ${RANK} \
             --data_path ${data_path} \
             --finetune ${finetune_pth} \
             --output_dir ${output_dir} \
             --model vit_large_patch16 \
             --epochs ${train_epochs} \
             --world_size 8 \
             --batch_size ${batch_size} \
             --num_workers ${KERNEL_NUM} \
             --blr 1e-3 \
             --layer_decay 0.65 \
             --weight_decay 0.05 \
             --drop_path 0.2 \
             --mixup 0.8 \
             --cutmix 1.0 \
             --reprob 0.25 \
             --dist_eval \
             --amp \
             --loss_scale 2048 \
             > ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &
done
wait


##################获取训练数据################
# 训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

# 训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'acc'

# 结果打印，不需要修改
echo "------------------ Final result ------------------"
# 输出性能FPS，需要模型审视修改
grep "train_one_epoch FPS:" ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | awk '{print $4}' >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_${CaseName}_fps.log
FPS=`cat ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${CaseName}_fps.log | awk '{a+=$1} END {if (NR != 0) printf("%.3f",a/NR)}'`
# 打印，不需要修改
echo "Final Performance images/sec : $FPS"

# 输出训练精度,需要模型审视修改
train_accuracy=`grep -a '* Acc@1'  ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk 'END {print}'|awk -F "Acc@1" '{print $NF}'|awk -F " " '{print $1}'`
# 打印，不需要修改
echo "Final Train Accuracy : ${train_accuracy}"
echo "E2E Training Duration sec : $e2e_time"

# 性能看护结果汇总
# 获取性能数据，不需要修改
# 吞吐量
ActualFPS=${FPS}
# 单迭代训练时长
TrainingTime=`awk 'BEGIN{printf "%.2f\n", '${batch_size}'*1000*8/'${FPS}'}'`

# 从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
grep Epoch: ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|grep -ve Test -ve Total|awk -F "loss:" '{print $NF}'|awk -F '[()]' '{print $2}' >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt

# 最后一个迭代loss值，不需要修改
ActualLoss=`awk 'END {print}'  ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt`

# 关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" >  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainAccuracy = ${train_accuracy}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log