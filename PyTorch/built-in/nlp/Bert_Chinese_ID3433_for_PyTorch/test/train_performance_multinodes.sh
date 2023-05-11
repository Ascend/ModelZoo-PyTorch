#!/bin/bash

cur_path=`pwd`
# 数据集路径,保持为空,不需要修改
data_path=""
device_id=0
# 每台服务器8卡训练卡
device_number=8
#基础参数，需要模型审视修改
#网络名称，同目录名称
Network="Bert_Chinese_ID3433_for_PyTorch"
#训练epoch
train_epochs=3
#训练batch_size 默认bert base batch size, 该参数外部可传入
batch_size=32
# 训练模型是bert base 还是bert large，默认bert base
model_size=base
warmup_ratio=0.0
weight_decay=0.0


#获取外部传参，可扩展
for para in $*
do
    if [[ $para == --data_path* ]];then
      data_path=`echo ${para#*=}`
    elif [[ $para == --batch_size* ]];then
      batch_size=`echo ${para#*=}`
    elif [[ $para == --nnodes* ]];then
      nnodes=`echo ${para#*=}`
    elif [[ $para == --device_number* ]];then
      device_number=`echo ${para#*=}`
    elif [[ $para == --node_rank* ]];then
      node_rank=`echo ${para#*=}`
    elif [[ $para == --master_addr* ]];then
      master_addr=`echo ${para#*=}`
    elif [[ $para == --master_port* ]];then
      master_port=`echo ${para#*=}`
    elif [[ $para == --model_size* ]];then
      model_size=`echo ${para#*=}`
    elif [[ $para == --warmup_ratio* ]];then
      warmup_ratio=`echo ${para#*=}`
    elif [[ $para == --weight_decay* ]];then
      weight_decay=`echo ${para#*=}`
    elif [[ $para == --conda_name* ]];then
        conda_name=`echo ${para#*=}`
        source set_conda.sh
        source activate $conda_name
    fi
done

#判断是否使用conda环境
if [[ $conda_name != "" ]];then
   cp -r $data_path/* $cur_path/../
   data_path=$data_path/'train_huawei.txt'
fi

#校验是否传入data_path,不需要修改
if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
    exit 1
fi

#集合通信参数,不需要修改
export RANK_SIZE=$((nnodes * device_number))
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

if [ $ASCEND_DEVICE_ID ];then
    echo "device id is ${ASCEND_DEVICE_ID}"
elif [ ${device_id} ];then
    export ASCEND_DEVICE_ID=${device_id}
    echo "device id is ${ASCEND_DEVICE_ID}"
else
    "[Error] device id must be config"
    exit 1
fi

#################创建日志输出目录，不需要修改#################
if [ -d ${test_path_dir}/output/${ASCEND_DEVICE_ID} ];then
    rm -rf ${test_path_dir}/output/${ASCEND_DEVICE_ID}
    mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID
else
    mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID
fi

#训练开始时间，不需要修改
start_time=$(date +%s)
# 非平台场景时source 环境变量
check_etp_flag=`env | grep etp_running_flag`
etp_flag=`echo ${check_etp_flag#*=}`
if [ x"${etp_flag}" != x"true" ];then
    source ${test_path_dir}/env_npu.sh
fi

nohup python3 -m torch.distributed.launch --nnodes=${nnodes} --node_rank=${node_rank} --nproc_per_node ${device_number} --master_addr ${master_addr} --master_port ${master_port} run_mlm.py \
        --model_type bert \
        --config_name ./bert-${model_size}-chinese/config.json \
        --tokenizer_name ./bert-${model_size}-chinese \
        --max_seq_length 512 \
        --train_file ${data_path} \
        --eval_metric_path ./accuracy.py \
        --line_by_line \
        --learning_rate 8e-05 \
        --pad_to_max_length \
        --remove_unused_columns false \
        --save_steps 5000 \
        --dataloader_num_workers 4 \
        --use_combine_ddp \
        --num_train_epochs ${train_epochs} \
        --overwrite_output_dir \
        --per_device_train_batch_size ${batch_size} \
        --per_device_eval_batch_size ${batch_size} \
        --do_train \
        --dataloader_drop_last true \
        --do_eval \
        --eval_accumulation_steps 100 \
        --fp16 \
        --warmup_ratio ${warmup_ratio} \
        --weight_decay ${weight_decay} \
        --fp16_opt_level O2 \
        --loss_scale 8192 \
        --use_combine_grad \
        --optim adamw_apex_fused_npu \
        --distributed_process_group_timeout 5400 \
        --output_dir ./output > ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &
wait

#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

self_ip=$(hostname -I |awk '{print $1}')

if [ x"${self_ip}" == x"${master_addr}" ]; then

    #结果打印，不需要修改
    echo "------------------ Final result ------------------"
    #输出性能FPS，需要模型审视修改
    FPS=`grep "train_samples_per_second ="  ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | awk 'END {print $3}'`

    #打印，不需要修改
    echo "Final Performance images/sec : $FPS"

    #输出训练精度,需要模型审视修改
    train_accuracy=`grep "eval_accuracy" ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | awk 'END {print $3}'`

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
    grep "{'loss" ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log |awk -F "{'loss" '{print $2}'|awk '{print $2}' | awk -F "," '{print $1}' > ${test_path_dir}/output/${ASCEND_DEVICE_ID}//train_${CaseName}_loss.txt
    #最后一个迭代loss值，不需要修改
    ActualLoss=`awk 'END {print}' ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${CaseName}_loss.txt`

    #关键信息打印到${CaseName}.log中，不需要修改
    echo "Network = ${Network}" > ${test_path_dir}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
    echo "RankSize = ${RANK_SIZE}" >> ${test_path_dir}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
    echo "BatchSize = ${BatchSize}" >> ${test_path_dir}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
    echo "DeviceType = ${DeviceType}" >> ${test_path_dir}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
    echo "CaseName = ${CaseName}" >> ${test_path_dir}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
    echo "ActualFPS = ${ActualFPS}" >> ${test_path_dir}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
    echo "TrainingTime = ${TrainingTime}" >> ${test_path_dir}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
    echo "ActualLoss = ${ActualLoss}" >> ${test_path_dir}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
    echo "E2ETrainingTime = ${e2e_time}" >> ${test_path_dir}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
    echo "TrainAccuracy = ${train_accuracy}">> ${test_path_dir}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
fi