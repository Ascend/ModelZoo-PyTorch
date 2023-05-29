#!/bin/bash

################基础配置参数##################
arch_network="RCAN" # 网络结构选择，可以不用指定，默认即可
batch_size=160 # 训练batch_size，多卡可以设置为1280，单卡设置为160
num_epochs=2 # 训练网络轮数，可以不用指定，默认即可
device="npu" # 训练设备选择 "npu" 或者 "gpu" ，可以不用指定，默认即可
train_dataset_dir="/root/dataset_zzq/train_setDVI2k_augsmall/" # 训练集路径
test_dataset_dir="/root/dataset_zzq/0_Set5/" # 测试集路径
outputs_dir="/root/RCAN_test_20210908/" # 输出保存路径
finetuning_checkpoint_path="/root/RCAN_test_20210908/X2/model_best_amp.pth"
amp="--amp" # 是否使用amp进行训练，可以不用指定，默认即可
scale=2  # 超分辨率放大倍数，可以不用指定，默认即可
device_id=0 # 设备编号
lr=1e-4 #学习率
Rank_Size=1 # 卡的数量

################接收外部输入配置参数##################
for para in $*
do
    if [[ $para == --arch_network* ]];then
        arch_network=`echo ${para#*=}`
    elif [[ $para == --scale* ]];then
        scale=`echo ${para#*=}`
    elif [[ $para == --batch_size* ]];then
        batch_size=`echo ${para#*=}`
    elif [[ $para == --num_epochs* ]];then
        num_epochs=`echo ${para#*=}`
    elif [[ $para == --device* ]];then
        device=`echo ${para#*=}`
    elif [[ $para == --train_dataset_dir* ]];then
        train_dataset_dir=`echo ${para#*=}`
    elif [[ $para == --test_dataset_dir* ]];then
        test_dataset_dir=`echo ${para#*=}`
    elif [[ $para == --outputs_dir* ]];then
        outputs_dir=`echo ${para#*=}`
    elif [[ $para == --lr* ]];then
        lr=`echo ${para#*=}`
    elif [[ $para == --device_id* ]];then
        device_id=`echo ${para#*=}`
    elif [[ $para == --finetuning_checkpoint_path* ]];then
        finetuning_checkpoint_path=`echo ${para#*=}`
    fi
done

#################创建日志输出目录，不需要修改#################
if [ -d ${outputs_dir}/X${scale}/ ];then
    log_path=${outputs_dir}/X${scale}/train.log
else
    mkdir -p ${outputs_dir}/X${scale}/
    log_path=${outputs_dir}/X${scale}/train.log
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
start_time=$(date +%s)

nohup python3 -u ../main.py  --arch ${arch_network} \
                          --batch_size ${batch_size} \
                          --num_epochs ${num_epochs} \
                          --device ${device} \
                          --train_dataset_dir ${train_dataset_dir} \
                          --test_dataset_dir ${test_dataset_dir} \
                          --outputs_dir ${outputs_dir} \
                          --scale ${scale} \
                          ${amp} \
                          --num_epochs ${num_epochs} \
                          --device_id ${device_id} \
                          --iffinetuning \
                          --finetuning_checkpoint_path ${finetuning_checkpoint_path}>> ${log_path} 2>&1 & 

wait
##################获取训练数据################

# 训练结束时间
end_time=$(date +%s)
e2e_time=$(($end_time-$start_time))
# 结果打印
echo "------------------ Final result ------------------"
TrainingTime=`grep -a "This Epoch's whole time:" ${log_path}|awk -F "This Epoch's whole time:" '{print $NF}' | awk -F " " '{print $1}' | awk 'END {print}'`
echo "Time for one epoch: ${TrainingTime}"
# 输出性能FPS
FPS=`grep -a 'FPS:' ${log_path}|awk -F "FPS:" '{print $NF}' | awk -F " " '{print $1}' | awk 'END {print}'`
echo "Final Performance images/sec: $FPS"
# 输出训练精度
train_accuracy=`grep -a 'The Best PSNR is' ${log_path}|awk -F "The Best PSNR is" '{print $NF}' | awk -F " " '{print $1}' | awk 'END {print}'`

if [[ ! $train_accuracy ]];then
train_accuracy=`grep -a 'PSNR:' ${log_path}|awk -F "PSNR:" '{print $NF}' | awk -F " " '{print $1}' | awk 'END {print}'`
fi
echo "Final Val PSNR: ${train_accuracy}"
# 输出训练loss
ActualLoss=`grep -a 'Loss:' ${log_path}|awk -F "Loss:" '{print $NF}' | awk -F " " '{print $1}' | awk 'END {print}'`
echo "Final loss: ${ActualLoss}"
echo "E2E Training Duration sec : $e2e_time"

# 性能看护结果汇总
DeviceType=`uname -m`
CaseName=${arch_network}_bs${batch_size}_${Rank_Size}'p'_'perf'

#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${arch_network}" >  ${outputs_dir}/X${scale}/${CaseName}.log
echo "RankSize = ${Rank_Size}" >>  ${outputs_dir}/X${scale}/${CaseName}.log
echo "BatchSize = ${batch_size}" >>  ${outputs_dir}/X${scale}/${CaseName}.log
echo "DeviceType = ${DeviceType}" >>  ${outputs_dir}/X${scale}/${CaseName}.log
echo "CaseName = ${CaseName}" >>  ${outputs_dir}/X${scale}/${CaseName}.log
echo "ActualFPS = ${FPS}" >>  ${outputs_dir}/X${scale}/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >>  ${outputs_dir}/X${scale}/${CaseName}.log
echo "ValPSNR = ${train_accuracy}" >> ${outputs_dir}/X${scale}/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >>  ${outputs_dir}/X${scale}/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >>  ${outputs_dir}/X${scale}/${CaseName}.log


exit



