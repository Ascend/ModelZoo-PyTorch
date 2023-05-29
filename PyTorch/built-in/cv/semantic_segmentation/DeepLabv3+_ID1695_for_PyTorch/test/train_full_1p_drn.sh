#!/bin/bash

# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
################basic configuration parameters##################
# required fields: Network batch_size RANK_SIZE
# network name, same as directory name
Network="DeepLabv3+_ID1695_for_PyTorch"
# training batch_size
batch_size=8
# number of npu cards used for training
export RANK_SIZE=1
# data set path, keep it empty, no need to modif
data_path=""
dataset="pascal"
num_classes=21

resume=""
ft=""

# training epoch
train_epochs=100
# Specifies the NPU device card ID used for training
device_id=0
# learning rate
learning_rate=0.001

# Parameter verification, data_path is a required parameter, 
# and the addition or deletion of other parameters is determined by the model itself;
# The parameters added here need to be defined and assigned above
for para in $*
do
    if [[ $para == --device_id* ]];then
        device_id=`echo ${para#*=}`
    elif [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    elif [[ $para == --more_path1* ]];then
        more_path1=`echo ${para#*=}`
    elif [[ $para == --precision_mode* ]];then
        precision_mode=`echo ${para#*=}`
    elif [[ $para == --train_epochs* ]];then
        train_epochs=`echo ${para#*=}`
    elif [[ $para == --backbone* ]];then
        backbone=`echo ${para#*=}`
    elif [[ $para == --checkname* ]];then
        checkname=`echo ${para#*=}`
    elif [[ $para == --dataset* ]];then
        dataset=`echo ${para#*=}`
    elif [[ $para == --num_classes* ]];then
        num_classes=`echo ${para#*=}`
    elif [[ $para == --ckpt_name* ]];then
        ckpt_name=`echo ${para#*=}`
    elif [[ $para == --resume* ]];then
        resume=`echo ${para#*=}`
        ft='--ft'
    fi
done

# verify whether data_path is passed in, no need to modify
if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
    exit 1
fi

# Verify that device is specified_ id, 
# dynamic allocation of device_id and manually specified device_ id,
# which does not need to be modified here
if [ $ASCEND_DEVICE_ID ];then
    echo "device id is ${ASCEND_DEVICE_ID}"
elif [ ${device_id} ];then
    export ASCEND_DEVICE_ID=${device_id}
    echo "device id is ${ASCEND_DEVICE_ID}"
else
    "[Error] device id must be config"
    exit 1
fi



###############Specify the training script execution path###############
# cd to the same level directory as the test folder to execute scripts to improve compatibility; 
# test_ path_ dir is the path containing the test folder
cur_path=`pwd`
cur_path_last_dirname=${cur_path##*/}
if [ x"${cur_path_last_dirname}" == x"test" ];then
    test_path_dir=${cur_path}
    cd ..
    cur_path=`pwd`
else
    test_path_dir=${cur_path}/test
fi

# The data set path is quoted; Change the dataset path in mypath.py
data_path=\"${data_path}\"
sed -i 's#dataset_path=.*$#dataset_path='$data_path'#' ./mypath.py
# cp pth file
if [ -f ${more_path1}/${ckpt_name} ]; then
    echo "pth file exists"
    mkdir -p ~/.cache/torch/checkpoints
    cp ${more_path1}/${ckpt_name} ~/.cache/torch/checkpoints/${ckpt_name}
fi

#################Create log output directory ,no modification required#################
if [ -d ${test_path_dir}/output/${ASCEND_DEVICE_ID} ];then
    rm -rf ${test_path_dir}/output/${ASCEND_DEVICE_ID}
    mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID
else
    mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID
fi


#################Start training script#################
# Training start time, no need to modify
start_time=$(date +%s)

# Source environment variable in non platform scenarios
check_etp_flag=`env | grep etp_running_flag`
etp_flag=`echo ${check_etp_flag#*=}`
if [ x"${etp_flag}" != x"true" ];then
    source ${test_path_dir}/env_npu.sh
fi

nohup python3 train.py \
    --backbone ${backbone} \
    --lr ${learning_rate} \
    --workers 64 \
    --epochs ${train_epochs} \
    --batch-size ${batch_size} \
    --device_id ${ASCEND_DEVICE_ID} \
    --checkname ${checkname} \
    --eval-interval 1 \
    --num_classes ${num_classes} \
    --resume "${resume}" \
    ${ft} \
    --dataset ${dataset} > ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &

wait

# Training end time, no need to modify
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'acc'

# The result is printed without modification
echo "------------------ Final result ------------------"
# Output performance FPS needs model review and modification
grep 'Epoch' ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | grep 'FPS' | awk '{print $7}' >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_${CaseName}_fps.txt
FPS=`cat ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_${CaseName}_fps.txt | awk '{a+=$1} END {if (NR != 0) printf("%.3f",a/NR)}'`
# Print without modification
echo "Final Performance images/sec : $FPS"

# The output training accuracy requires model review and modification
train_accuracy=`grep 'val_mIoU' ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk -F "," '{print $3}' | awk -F ":" '{print $2}'| sort | awk 'END {print}'`
# Print without modification
echo "Final Train Accuracy : ${train_accuracy}"
echo "E2E Training Duration sec : $e2e_time"

# Get performance data without modification
# throughput
ActualFPS=${FPS}
# Single iteration training duration
TrainingTime=`awk 'BEGIN{printf "%.2f\n", '${batch_size}'*1000/'${FPS}'}'`

# from train_$ASCEND_DEVICE_ID.log fetch Loss to train_${CaseName}_loss.txt，reviewed according to the model
grep 'train_loss' ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log | awk -F ":" '{print $2}' >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt

# The loss of the last iteration, does not need to be modified
ActualLoss=`cat ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt | awk '{a+=$1} END {if (NR != 0) printf("%.3f",a/NR)}'`

# print key information to ${CaseName}.log，does not need to be modified
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