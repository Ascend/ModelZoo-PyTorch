#!/bin/bash

################基础配置参数，需要模型审视修改##################
# 必选字段(必须在此处定义的参数): Network batch_size RANK_SIZE
# 网络名称，同目录名称
Network="TNT_for_PyTorch"
# 训练batch_size
batch_size=1024
# 训练使用的npu卡数
export RANK_SIZE=8
# 数据集路径
data_path=""

# 训练epoch
train_epochs=310

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

KERNEL_NUM=$(($(nproc)/8))

rm -rf 8p_full.log

for((RANK_ID=0;RANK_ID<RANK_SIZE;RANK_ID++))
do
    export RANK_ID=$RANK_ID
    if [ $(uname -m) = "aarch64" ]
    then
		PID_START=$((KERNEL_NUM * RANK_ID))
		PID_END=$((PID_START + KERNEL_NUM - 1))
		nohup taskset -c $PID_START-$PID_END python3 train.py ${data_path} \
			--model tnt_s_patch16_224 \
			--sched cosine \
			--epochs 300 \
			--opt adamw \
			-j 8 \
			--warmup-lr 1e-6 \
			--mixup .8 \
			--cutmix 1.0 \
			--model-ema \
            --model-ema-decay 0.99996 \
			--aa rand-m9-mstd0.5-inc1 \
			--color-jitter 0.4 \
			--warmup-epochs 5 \
			--opt-eps 1e-8 \
			--repeated-aug \
			--remode pixel \
			--reprob 0.25 \
			--amp \
			--lr 1e-3 \
			--weight-decay .05 \
			--drop 0 \
			--drop-path .1 \
			-b 128 \
			--addr $(hostname -I |awk '{print $1}') \
			--output ./train_cache \
			--local_rank $RANK_ID > 8p_full.log 2>&1 &
    else
		nohup python3 -m torch.distributed.launch --nproc_per_node=8 train.py ${data_path} \
			--model tnt_s_patch16_224 \
			--sched cosine \
			--epochs 300 \
			--opt adamw \
			-j 8 \
			--warmup-lr 1e-6 \
			--mixup .8 \
			--cutmix 1.0 \
			--model-ema \
            --model-ema-decay 0.99996 \
			--aa rand-m9-mstd0.5-inc1 \
			--color-jitter 0.4 \
			--warmup-epochs 5 \
			--opt-eps 1e-8 \
			--repeated-aug \
			--remode pixel \
			--reprob 0.25 \
			--amp \
			--lr 1e-3 \
			--weight-decay .05 \
			--drop 0 \
			--drop-path .1 \
			-b 128 \
			--addr $(hostname -I |awk '{print $1}') \
			--output ./train_cache > 8p_full.log 2>&1 &
    fi
done

wait

cp 8p_full.log ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log


##################获取训练数据################
#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
FPS=`grep -a 'fps'  ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk -F " " '{print $3}'|awk 'END {print}'`
#打印，不需要修改
echo "Final Performance images/sec : $FPS"

#输出训练精度,需要模型审视修改
train_accuracy=`grep -a ".pth.tar'," ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk -F " " '{print $2}'|awk -F ")" '{print $1}'|awk 'BEGIN {max = 0} {if ($1+0 > max+0) max=$1} END {print max}'`
#打印，不需要修改
echo "Final Train Accuracy : ${train_accuracy}"
echo "E2E Training Duration sec : $e2e_time"

#性能看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'acc'

##获取性能数据，不需要修改
#吞吐量
ActualFPS=${FPS}
#单迭代训练时长
TrainingTime=`awk 'BEGIN{printf "%.2f\n", '${batch_size}'*1000/'${FPS}'}'`

#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
grep Train: ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|grep -v Test|awk -F "Loss:" '{print $NF}'|awk -F " " '{print $1}' >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt

#最后一个迭代loss值，不需要修改
ActualLoss=`awk 'END {print}'  ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt`

#关键信息打印到${CaseName}.log中，不需要修改
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