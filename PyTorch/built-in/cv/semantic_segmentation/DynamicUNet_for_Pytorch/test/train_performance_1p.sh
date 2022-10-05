#!/bin/bash

#网络名称，同目录名称
Network="DynamicUNet_ID4080_for_Pytorch"
batch_size=32

# 数据集路径,保持为空,不需要修改
data_path=""

# 预训练模型路径
ckpt_path=""

#参数校验，不需要修改
for para in $*
do
    if [[ $para == --ckpt_path* ]];then
        ckpt_path=`echo ${para#*=}`
    elif [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    fi
done

#校验是否传入data_path,不需要修改
if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
    exit 1
fi

if [[ $ckpt_path == "" ]];then
	pretrained_model="./"
else
	pretrained_model=${ckpt_path}/resnet50-19c8e357.pth 
fi 


###############指定训练脚本执行路径###############
# cd到与test文件夹同层级目录下执行脚本，提高兼容性；test_path_dir为包含test文件夹的路径
cur_path=$(pwd)
cur_path_last_dirname=${cur_path##*/}
if [ x"${cur_path_last_dirname}" == x"test" ]; then
	test_path_dir=${cur_path}
	cd ..
	cur_path=$(pwd)
else
	test_path_dir=${cur_path}/test
fi

ASCEND_DEVICE_ID=0

#################创建日志输出目录，不需要修改#################
if [ -d ${test_path_dir}/output/${ASCEND_DEVICE_ID} ]; then
	rm -rf ${test_path_dir}/output/${ASCEND_DEVICE_ID}
	mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID
else
	mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID
fi

#训练开始时间，不需要修改
start_time=$(date +%s)
# 非平台场景时source 环境变量
check_etp_flag=$(env | grep etp_running_flag)
etp_flag=$(echo ${check_etp_flag#*=})
if [ x"${etp_flag}" != x"true" ]; then
	source ${test_path_dir}/env_npu.sh
fi

# train
export PYTHONPATH=./awesome-semantic-segmentation-pytorch:$PYTHONPATH

nohup python3 -u runner.py \
--model dynamicunet --amp \
--dataset pascal_voc --dataset-path ${data_path} \
--lr 0.0001 --epochs 50 --worker 8 \
--log-iter 1 --val-epoch 5 --perf-only \
--pretrained ${pretrained_model} \
>${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &

wait

#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(($end_time - $start_time))

#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
FPS=$(grep "FPS:" ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | awk 'END {print}' | awk -F "FPS:" '{print $2}' | awk -F " " '{print $1}')

#打印，不需要修改
echo "Final Performance images/sec : $FPS"

#输出训练精度,需要模型审视修改
train_accuracy=$(grep "mIoU" ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | awk 'END {print $7}')

#打印，不需要修改
echo "Final Train Accuracy : ${train_accuracy}"
echo "E2E Training Duration sec : $e2e_time"

#性能看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=$(uname -m)
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'acc'

##获取性能数据，不需要修改
#吞吐量
ActualFPS=${FPS}
#单迭代训练时长
TrainingTime=$(awk 'BEGIN{printf "%.2f\n", '${batch_size}'*1000/'${FPS}'}')

#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
grep "Loss" ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | awk 'END {print}' | awk -F "(" '{print $5}' | awk -F ")" '{print $1}' >${test_path_dir}/output/${ASCEND_DEVICE_ID}//train_${CaseName}_loss.txt
#最后一个迭代loss值，不需要修改
ActualLoss=$(awk 'END {print}' ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${CaseName}_loss.txt)

#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" >${test_path_dir}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >>${test_path_dir}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "BatchSize = ${BatchSize}" >>${test_path_dir}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "DeviceType = ${DeviceType}" >>${test_path_dir}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "CaseName = ${CaseName}" >>${test_path_dir}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >>${test_path_dir}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >>${test_path_dir}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >>${test_path_dir}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >>${test_path_dir}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "TrainAccuracy = ${train_accuracy}" >>${test_path_dir}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
