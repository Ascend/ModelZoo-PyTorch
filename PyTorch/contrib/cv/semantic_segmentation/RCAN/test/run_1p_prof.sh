# BSD 3-Clause License
#
# Copyright (c) 2017 xxxx
# All rights reserved.
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ============================================================================


#!/bin/bash

################基础配置参数##################

arch_network="RCAN" # 网络结构选择，可以不用指定，默认即可
batch_size=160 # 训练batch_size，多卡可以设置为1280，单卡设置为160
device="npu" # 训练设备选择 "npu" 或者 "gpu" ，可以不用指定，默认即可
train_dataset_dir="/root/dataset_zzq/train_setDVI2k_augsmall/" # 训练集路径
test_dataset_dir="/root/dataset_zzq/0_Set5/" # 测试集路径
outputs_dir="/root/RCAN_test_20210908/" # 输出保存路径
amp="--amp" # 是否使用amp进行训练，可以不用指定，默认即可
scale=2  # 超分辨率放大倍数，可以不用指定，默认即可
device_id=0 # 设备编号
RANK_SIZE=1
################接收外部输入配置参数##################
for para in $*
do
    if [[ $para == --arch_network* ]];then
        arch_network=`echo ${para#*=}`
    elif [[ $para == --scale* ]];then
        scale=`echo ${para#*=}`
    elif [[ $para == --batch_size* ]];then
        batch_size=`echo ${para#*=}`
    elif [[ $para == --device* ]];then
        device=`echo ${para#*=}`
    elif [[ $para == --train_dataset_dir* ]];then
        train_dataset_dir=`echo ${para#*=}`
    elif [[ $para == --test_dataset_dir* ]];then
        test_dataset_dir=`echo ${para#*=}`
    elif [[ $para == --outputs_dir* ]];then
        outputs_dir=`echo ${para#*=}`
    elif [[ $para == --device_id* ]];then
        device_id=`echo ${para#*=}`
    fi
done

#################创建日志输出目录，不需要修改#################
if [ -d ${outputs_dir}/X${scale}/${device} ];then
    log_path=${outputs_dir}/X${scale}/${device}/prof.log
else
    mkdir -p ${outputs_dir}/X${scale}/${device}/
    log_path=${outputs_dir}/X${scale}/${device}/prof.log
fi

#################激活环境，修改环境变量#################
if [ ${device} == "npu" ];then
    check_etp_flag=`env | grep etp_running_flag`
    etp_flag=`echo ${check_etp_flag#*=}`
    if [ x"${etp_flag}" != x"true" ];then
        source env_npu.sh
    fi
    export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
else
    source activate pt-1.5
fi


#################启动训练脚本#################
# 训练开始时间
export ASCEND_GLOBAL_LOG_LEVEL=3
export TASK_QUEUE_ENABLE=1
start_time=$(date +%s)

nohup python3 -u ../main_prof.py  --arch ${arch_network} \
                          --batch_size ${batch_size} \
                          --train_dataset_dir  ${train_dataset_dir}  \
                          --num_epochs 1 \
                          --outputs_dir ${train_dataset_dir} \
                          --test_dataset_dir ${outputs_dir} \
                          --amp \
                          --device ${device} \
                          --device_id ${device_id}  > ${log_path} 2>&1 & 

wait

##################获取训练数据################
# 训练结束时间
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))
# 结果打印
echo "------------------ Final result ------------------"
FPS=`grep -a 'FPS'  ${log_path}|awk -F " " '{print $11}'|awk 'END {print}'`
#打印，不需要修改
echo "Final Performance images/sec : $FPS"

#输出训练精度,需要模型审视修改
train_accuracy=`grep -a '* Acc@1'  ${log_path}|awk 'END {print}'|awk -F "Acc@1" '{print $NF}'|awk -F " " '{print $1}'`
#打印，不需要修改
echo "Final Train Accuracy : ${train_accuracy}"
echo "E2E Training Duration sec : $e2e_time"


# 性能看护结果汇总
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${arch_network}_bs${batch_size}_${RANK_SIZE}'p'_'perf'
ActualFPS=${FPS}
#单迭代训练时长
ActualLoss=`awk 'END {print}'  ${log_path}`

#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${arch_network}" >  ${outputs_dir}/X${scale}/${device}/${CaseName}.log
echo "RankSize = ${Rank_Size}" >>  ${outputs_dir}/X${scale}/${device}/${CaseName}.log
echo "BatchSize = ${batch_size}" >>  ${outputs_dir}/X${scale}/${device}/${CaseName}.log
echo "DeviceType = ${DeviceType}" >>  ${outputs_dir}/X${scale}/${device}/${CaseName}.log
echo "CaseName = ${CaseName}" >>  ${outputs_dir}/X${scale}/${device}/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >>  ${outputs_dir}/X${scale}/${device}/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >>  ${outputs_dir}/X${scale}/${device}/${CaseName}.log
echo "ValPSNR = ${train_accuracy}" >> ${outputs_dir}/X${scale}/${device}/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >>  ${outputs_dir}/X${scale}/${device}/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >>  ${outputs_dir}/X${scale}/${device}/${CaseName}.log
exit
