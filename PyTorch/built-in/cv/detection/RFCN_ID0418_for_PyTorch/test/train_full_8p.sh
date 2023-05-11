#!/bin/bash

################基础配置参数，需要模型审视修改##################
# 必选字段(必须在此处定义的参数): Network batch_size RANK_SIZE
# 网络名称，同目录名称
Network="RFCN_ID0418_for_PyTorch"
# 训练batch_size
batch_size=32
# 训练使用的npu卡数
export RANK_SIZE=8
# 数据集路径,保持为空,不需要修改
data_path=""
# 训练epoch
train_epochs=20
# 指定训练所使用的npu device卡id
device_id=0
# 学习率
learning_rate=0.008
# 加载数据进程数
workers=8
#预训练模型路径
pretrained_model_path="/npu/rfcn_pretrained_model/"

# 参数校验，data_path为必传参数，其他参数的增删由模型自身决定；此处新增参数需在上面有定义并赋值
for para in $*
do
    if [[ $para == --device_id* ]];then
        device_id=`echo ${para#*=}`
    elif [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    elif [[ $para == --precision_mode* ]];then
        precision_mode=`echo ${para#*=}`
    elif [[ $para == --ci_cp* ]];then
        ci_cp=`echo ${para#*=}`
    fi
done

if [[ $ci_cp == "1" ]];then
    cp -r $data_path ${data_path}_bak
fi

PREC=""
if [[ $precision_mode == "amp" ]];then
    PREC="--amp"
fi
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

# 新建数据集及与训练权重放置目录，并建立软连接
mkdir -p data
cd data
ln -s ${data_path}/VOCdevkit2007 VOCdevkit2007
ln -s ${data_path} pretrained_model
cd ..

#非平台场景时source 环境变量
check_etp_flag=`env | grep etp_running_flag`
etp_flag=`echo ${check_etp_flag#*=}`
if [ x"${etp_flag}" != x"true" ];then
    source  ${test_path_dir}/env_npu.sh
fi

#################创建日志输出目录，不需要修改#################
KERNEL_NUM=$(($(nproc)/8))
for i in $(seq 0 7)
do
ASCEND_DEVICE_ID=$i
if [ -d ${test_path_dir}/output/$ASCEND_DEVICE_ID ];then
    rm -rf ${test_path_dir}/output/$ASCEND_DEVICE_ID
    mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID
else
    mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID
fi
#################启动训练脚本#################
#训练开始时间，不需要修改
start_time=$(date +%s)

PID_START=$((KERNEL_NUM * i))
PID_END=$((PID_START + KERNEL_NUM - 1))
taskset -c $PID_START-$PID_END nohup python3 -u ./trainval_net_8p.py \
    --net=res101 \
    --nw=${workers} \
    --lr=${learning_rate} \
    --lr_decay_step=8  \
    --disp_interval=1 \
    --device=npu \
    --epochs=${train_epochs} \
    --bs=${batch_size} \
    --npu_id="npu:${ASCEND_DEVICE_ID}" \
    --local_rank=${ASCEND_DEVICE_ID} \
    --amp \
    --opt_level=O1 \
    --loss_scale=1024.0 > ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &
done
wait

ASCEND_DEVICE_ID=0

nohup python3 ./test_net.py \
    --arch=rfcn \
    --dataset=pascal_voc \
    --net=res101 \
    --cfg=cfg/res101.yml \
    --checksession 1 \
    --checkepoch ${train_epochs} \
    --checkpoint 312 \
    --device=npu \
    --npu_id="npu:${ASCEND_DEVICE_ID}" \
    --amp \
    --opt_level=O1 \
    --loss_scale=1024. > ${test_path_dir}/output/${ASCEND_DEVICE_ID}/test_${ASCEND_DEVICE_ID}.log 2>&1 &
wait


# 训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

# 结果打印，不需要修改
echo "------------------ Final result ------------------"
# 输出性能FPS，需要模型审视修改
FPS=`grep -a 'FPS'  ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk -F " " '{print $NF}'|awk 'END {print}'`
# 打印，不需要修改
echo "Final Performance images/sec : $FPS"

# 输出训练精度,需要模型审视修改
train_accuracy=`grep -a 'Mean AP =' ${test_path_dir}/output/${ASCEND_DEVICE_ID}/test_${ASCEND_DEVICE_ID}.log|awk 'END {print}'|awk -F "Mean AP =" '{print $NF}'|awk -F " " '{print $1}'`
# 打印，不需要修改
echo "Final Train Accuracy : ${train_accuracy}"
echo "E2E Training Duration sec : $e2e_time"

# 性能看护结果汇总
# 训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'acc'

# 获取性能数据，不需要修改
# 吞吐量
ActualFPS=${FPS}
# 单迭代训练时长
TrainingTime=`awk 'BEGIN{printf "%.2f\n", '${batch_size}'*1000/'${FPS}'}'`

# 从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
grep loss: ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|grep -a 'session'|awk -F "loss: " '{print $NF}' | awk -F " " '{print $1}' |awk -F ',' '{print $1}' >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt

# 最后一个迭代loss值，不需要修改
ActualLoss=`awk 'END {print}' ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt`

# 关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" > ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainAccuracy = ${train_accuracy}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log

if [[ $ci_cp == "1" ]];then
    rm -rf $data_path
    mv ${data_path}_bak $data_path
fi
