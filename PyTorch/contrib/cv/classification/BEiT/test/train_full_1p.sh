#!/bin/bash

##################基础配置参数，需要模型审视修改##################
# 指定参数 --data_path=XXX
# 网络名称，同目录名称
Network="beit_base_patch16_224"
# 所选模型
model="beit_base_patch16_224"
# 训练batch_size
batch_size=64
#训练epoch
epochs=30
# 训练使用的npu卡数
RANK_SIZE=1
# 数据集类别数量
nb_classes=1000
# 数据集路径,保持为空,不需要修改
data_path=""
finetune="./checkpoints/beit_base_patch16_224_pt22k_ft22k.pth"
# 训练结果保存路径
output_dir="./train_result/"

# 帮助信息，不需要修改
if [[ $1 == --help || $1 == -h ]];then
    echo"usage:./train_performance_8P.sh <args>"
    echo " "
    echo "parameter explain:
    --model                    choose the training model              
    --nb_classes               numbers of data classes
    --data_path		           source data
    -h/--help		           show help message
    "
    exit 1
fi

#参数校验，不需要修改                   ***********************************?
for para in $*
do
    if [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    elif [[ $para == --finetune* ]];then
        finetune=`echo ${para#*=}`
    elif [[ $para == --nb_classes* ]];then
        nb_classes=`echo ${para#*=}`
    fi
done
echo "output_dir: $output_dir"
echo "data_path: $data_path"

#校验是否传入data_path,不需要修改
if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
    exit 1
fi
##################指定训练脚本执行路径##################            
# cd到与test文件同层级目录下执行脚本，提高兼容性；test_path_dir为包含test文件夹的路径
cur_path=`pwd`
cur_path_last_dirname=${cur_path##*/}
if [ x"${cur_path_last_dirname}" == x"test" ]; then
    test_path_dir=${cur_path}
    cd ..
    cur_path=`pwd`
else
    test_path_dir=${cur_path}/test
fi
echo "cur_path: ${cur_path}"
##################创建日志输出目录，根据模型审视##################              **********************************?
# 模型采用非循环方式启动多卡训练，创建日志输出目录如下；采用循环方式启动多卡训练的模型，在循环中创建日志输出目录，可参考CRNN模型
# 非循环方式下8卡训练日志输出路径中的ASCEND_DEVICE_ID默认为0，只是人为指定文件夹名称， 不涉及训练业务
ASCEND_DEVICE_ID=0
if [ -d ${test_path_dir}/output/$ASCEND_DEVICE_ID ];then
    rm -rf ${test_path_dir}/output/$ASCEND_DEVICE_ID
    mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID
else
    mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID
fi
echo "test_path_dir: ${test_path_dir}"
##################启动训练脚本##################
#训练开始时间，不需要修改
start_time=$(date +%s)
# source 环境变量
source ./test/env_npu.sh

OMP_NUM_THREADS=1 python -m torch.distributed.launch \
    --nproc_per_node=${RANK_SIZE} \
    run_class_finetuning_apex_npu.py \
    --model ${model} \
    --data_path ${data_path} \
    --finetune ${finetune} \
    --output_dir ${output_dir} \
    --batch_size ${batch_size} \
    --lr 2e-5 \
    --update_freq 1 \
    --warmup_epochs 5 \
    --epochs ${epochs} \
    --layer_decay 0.85 \
    --drop_path 0.1 \
    --weight_decay 1e-8 \
    --nb_classes ${nb_classes} \
    --amp \
    --device npu \
    --opt npufusedadamw > ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &
wait

##################获取训练数据##################
#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

#结果打印，不需要修改
echo "------------------ Final result ------------------"
# 输出性能FPS，需要模型审视修改
# Epoch: [2] Total time: 0:12:55 (0.3098 s / it)    sec/iterable
FPS=`grep -a 'Total time'  ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log| grep "Epoch" |awk -F "(" '{print $NF}'|awk -F ")" '{print $1}' | awk 'END {print}'`

# 打印，不需要修改
echo "Final Performance image/sec : $FPS"

# 输出训练精度,需要模型审视修改
# * Acc@1 85.249 Acc@5 97.607 loss 0.625
# Accuracy of the network on the 49677 test images: 85.2%
# train_accuracy1=`grep -a 'Acc@1' ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk 'END {print}'`
# train_accuracy2=`grep -a 'Accuracy of the network on the' ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk 'END {print}'`
train_accuracy=`grep -a 'Accuracy of the network on the' ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk 'END {print}'| awk -F " " '{print $NF}'`
# 打印，不需要修改
echo "Final Train Accuracy : ${train_accuracy}"
echo "E2E Training Duration sec : $e2e_time s"

#性能看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'acc'
echo "CaseName: $CaseName"
# 获取性能数据
# 吞吐量
ActualFPS=${FPS}
# 单迭代训练时长
TrainingTime=`grep -a 'Training time' ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log`
echo "TrainingTime: $TrainingTime"
# 从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
# Test:  [490/518]  eta: 0:00:05  loss: 0.4872 (0.6235)  acc1: 89.5833 (85.2936)  acc5: 98.9583 (97.6006)  time: 0.1970  data: 0.0001  max mem: 944
grep 'Test:  ' ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log | awk -F "]" '{print $2}' |awk -F "  " '{print $3}'  >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt
# 最后一个迭代loss值
ActualLoss=`awk 'END {print}' ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt`
# 关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" > ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainAccuracy = ${train_accuracy}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
