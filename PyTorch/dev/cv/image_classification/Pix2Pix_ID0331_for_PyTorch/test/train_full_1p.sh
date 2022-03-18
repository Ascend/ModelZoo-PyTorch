#!/bin/bash

#当前路径,不需要修改
cur_path=`pwd`

#集合通信参数,不需要修改

export RANK_SIZE=1
RANK_ID_START=0

# 数据集路径,保持为空,不需要修改
data_path=""


#基础参数，需要模型审视修改
#网络名称，同目录名称
Network="Pix2Pix_ID0331_for_PyTorch"
#训练epoch
train_epochs=200
#训练step
train_steps=506
#训练batch_size
batch_size=1
#学习率
learning_rate=8e-2

#维测参数，precision_mode需要模型审视修改
precision_mode="allow_fp32_to_fp16"
#维持参数，以下不需要修改
over_dump=False
data_dump_flag=False
data_dump_step="10"
profiling=False

# 帮助信息，不需要修改
if [[ $1 == --help || $1 == -h ]];then
    echo"usage:./train_performance_1P.sh <args>"
    echo " "
    echo "parameter explain:
    --precision_mode         precision mode(allow_fp32_to_fp16/force_fp16/must_keep_origin_dtype/allow_mix_precision)
    --over_dump              if or not over detection, default is False
    --data_dump_flag         data dump flag, default is False
    --data_dump_step         data dump step, default is 10
    --profiling              if or not profiling for performance debug, default is False
    --data_path              source data of training
    --max_step               # of step for training
    --learning_rate          learning rate
    --batch                  batch size
    --modeldir               model dir
    --save_interval          save interval for ckpt
    --loss_scale             enable loss scale ,default is False
    -h/--help                show help message
    "
    exit 1
fi

#参数校验，不需要修改
for para in $*
do
    if [[ $para == --precision_mode* ]];then
        precision_mode=`echo ${para#*=}`
    elif [[ $para == --over_dump* ]];then
        over_dump=`echo ${para#*=}`
        over_dump_path=${cur_path}/output/overflow_dump
        mkdir -p ${over_dump_path}
    elif [[ $para == --data_dump_flag* ]];then
        data_dump_flag=`echo ${para#*=}`
        data_dump_path=${cur_path}/output/data_dump
        mkdir -p ${data_dump_path}
    elif [[ $para == --data_dump_step* ]];then
        data_dump_step=`echo ${para#*=}`
    elif [[ $para == --profiling* ]];then
        profiling=`echo ${para#*=}`
        profiling_dump_path=${cur_path}/output/profiling
        mkdir -p ${profiling_dump_path}
    elif [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    elif [[ $para == --max_step* ]];then
        train_steps=`echo ${para#*=}`
    elif [[ $para == --learning_rate* ]];then
        learning_rate=`echo ${para#*=}`
    elif [[ $para == --batch* ]];then
        batch_size=`echo ${para#*=}`
    elif [[ $para == --modeldir* ]];then
        modeldir=`echo ${para#*=}`
    elif [[ $para == --save_interval* ]];then
        save_interval=`echo ${para#*=}`
    elif [[ $para == --loss_scale* ]];then
        loss_scale=`echo ${para#*=}`
    fi
done

#校验是否传入data_path,不需要修改
if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be config"
    exit 1
fi


#############执行训练#########################
#训练开始时间，不需要修改
start_time=$(date +%s)

cd $cur_path/../
#进入训练脚本目录，需要模型审视修改
for((RANK_ID=$RANK_ID_START;RANK_ID<$((RANK_SIZE+RANK_ID_START));RANK_ID++));
do
    #设置环境变量，不需要修改
    echo "Device ID: $ASCEND_DEVICE_ID"
    export RANK_ID=$RANK_ID

    #创建DeviceID输出目录，不需要修改
    if [ -d ${cur_path}/output/${ASCEND_DEVICE_ID} ];then
        rm -rf ${cur_path}/output/${ASCEND_DEVICE_ID}
        mkdir -p ${cur_path}/output/$ASCEND_DEVICE_ID/ckpt
    else
        mkdir -p ${cur_path}/output/$ASCEND_DEVICE_ID/ckpt
    fi

    #执行训练脚本，以下传参不需要修改，其他需要模型审视修改
    #--data_path, --model_dir, --precision_mode, --precision_mode, --over_dump, --over_dump_path，--data_dump_flag，--data_dump_step，--data_dump_path，--profiling，--profiling_dump_path，--autotune
    nohup python3 implementations/pix2pix/pix2pix.py --data_path=${data_path} --apex --loss_scale 128 --n_epochs=${train_epochs} > ${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &
if [ $? -ne 0 ];then
  exit 1
fi
done
wait


#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
#单步平均时间
TrainingTime=`grep "time: " $cur_path/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log |awk -F "time: " '{print $2}' |tail -n +4| awk -F "] ETA" '{print $1}'|awk '{sum+=$1}END {print"",sum/NR}'`
#FPS
FPS=`python3 -c "print(${batch_size}/${TrainingTime})"`
#打印，不需要修改
echo "Final Performance images/sec : $FPS"

#输出训练精度,需要模型审视修改
#train_accuracy=`grep "Accuracy:" $cur_path/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log | awk -F " " '{print $2}'`
#打印，不需要修改
#echo "Final Train Accuracy : ${train_accuracy}"
echo "E2E Training Duration sec : $e2e_time"

#性能看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'acc'

##获取性能数据，不需要修改
#吞吐量
ActualFPS=${FPS}

#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
grep "pixel: " $cur_path/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log | awk -F "pixel: " '{print $2}' | awk -F ", adv" '{print $1}' > $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}_loss.txt
#最后一个迭代loss值，不需要修改
ActualLoss=(`awk 'END {print $NF}' $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}_loss.txt`)

#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" > $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
