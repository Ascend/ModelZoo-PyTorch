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
batch_size=64
# number of npu cards used for training
export RANK_SIZE=8
RANK_ID_START=0
# data set path, keep it empty, no need to modify
data_path=""
dataset="pascal"
num_classes=21
resume=""
ft=""

# training epoch
train_epochs=100
# learning rate
learning_rate=0.008

# parameter verification, data_path is a required parameter
for para in $*
do
    if [[ $para == --train_epochs* ]];then
        train_epochs=`echo ${para#*=}`
    elif [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    elif [[ $para == --more_path1* ]];then
        more_path1=`echo ${para#*=}`
    elif [[ $para == --precision_mode* ]];then
        precision_mode=`echo ${para#*=}`
    elif [[ $para == --backbone* ]];then
        backbone=`echo ${para#*=}`
    elif [[ $para == --checkname* ]];then
        checkname=`echo ${para#*=}`
    elif [[ $para == --dataset* ]];then
        dataset=`echo ${para#*=}`
	elif [[ $para == --epochs* ]];then
        epochs=`echo ${para#*=}`	
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

###############specify the execution path of the training script###############
# test_path_dir is the path containing the test folder
cur_path=`pwd`
cur_path_last_dirname=${cur_path##*/}
if [ x"${cur_path_last_dirname}" == x"test" ];then
    test_path_dir=${cur_path}
    cd ..
    cur_path=`pwd`
else
    test_path_dir=${cur_path}/test
fi

# quote the data set path; change the data set path in mypath.py
data_path=\"${data_path}\"
sed -i 's#dataset_path=.*$#dataset_path='$data_path'#' ./mypath.py
# cp pth file
if [ -f ${more_path1}/${ckpt_name} ]; then
    echo "pth file exists"
    mkdir -p /root/.cache/torch/checkpoints
    cp ${more_path1}/${ckpt_name} /root/.cache/torch/checkpoints/${ckpt_name}
fi

#################start the training script#################
# training start time, no need to modify 
start_time=$(date +%s)

for((RANK_ID=$RANK_ID_START;RANK_ID<$((RANK_SIZE+RANK_ID_START));RANK_ID++));
do
    # set environment variables, no need to modify
    echo "Device ID: $RANK_ID"
    export RANK_ID=$RANK_ID
    export ASCEND_DEVICE_ID=$RANK_ID
    ASCEND_DEVICE_ID=$RANK_ID

    # create DeviceID output directory, no need to modify
    if [ -d ${test_path_dir}/output/${ASCEND_DEVICE_ID} ];then
        rm -rf ${test_path_dir}/output/${ASCEND_DEVICE_ID}
        mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID
    else
        mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID
    fi
    # source environment variables in non-platform scenarios
    check_etp_flag=`env | grep etp_running_flag`
    etp_flag=`echo ${check_etp_flag#*=}`
    if [ x"${etp_flag}" != x"true" ];then
        source ${test_path_dir}/env_npu.sh
    fi

    nohup python3.7train.py \
        --backbone ${backbone} \
        --lr ${learning_rate} \
        --workers 64 \
        --epochs ${train_epochs} \
        --batch-size ${batch_size} \
        --checkname ${checkname} \
        --eval-interval 1 \
        --dataset ${dataset} \
        --num_classes ${num_classes} \
        --resume "${resume}" \
        ${ft} \
        --multiprocessing_distributed > ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &
done
wait

# in the case of 8p, only 0 card (master node) has a complete log, so subsequent log extraction only involves 0 card 
ASCEND_DEVICE_ID=0

# training end time, no need to modify
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'acc'

# the result is printed without modification
echo "------------------ Final result ------------------"
# output performance FPS
grep 'Epoch' ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | grep 'FPS' | awk '{print $7}' >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_${CaseName}_fps.txt
FPS=`cat ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_${CaseName}_fps.txt | awk '{a+=$1} END {if (NR != 0) printf("%.3f",a/NR)}'`
# print, no need to modify
echo "Final Performance images/sec : $FPS"

# output training accuracy
train_accuracy=`grep 'val_mIoU' ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk -F "," '{print $3}' | awk -F ":" '{print $2}'| sort | awk 'END {print}'`
# print, no need to modify
echo "Final Train Accuracy : ${train_accuracy}"
echo "E2E Training Duration sec : $e2e_time"

# get performance data without modification 
# throughput 
ActualFPS=${FPS}
# single iteration training duration
TrainingTime=`awk 'BEGIN{printf "%.2f\n", '${batch_size}'*1000/'${FPS}'}'`

# extract Loss to train_${CaseName}_loss.txt
grep 'train_loss' ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log | awk -F ":" '{print $2}' >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt

# the last iterative loss value does not need to be modified
ActualLoss=`cat ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt | awk '{a+=$1} END {if (NR != 0) printf("%.3f",a/NR)}'`

# the key information is printed to ${CaseName}.log, no need to modify
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