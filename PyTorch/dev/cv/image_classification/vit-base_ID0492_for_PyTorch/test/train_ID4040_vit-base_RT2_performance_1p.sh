#!/bin/bash

#当前路径,不需要修改
cur_path=`pwd`

#集合通信参数,不需要修改

export RANK_SIZE=1
export JOB_ID=10087
RANK_ID_START=0

RANK_SIZE=1
# 数据集路径,保持为空,不需要修改
data_path="/npu/traindata/imagenet_pytorch/"

#设置默认日志级别,不需要修改
export ASCEND_GLOBAL_LOG_LEVEL=3

#基础参数，需要模型审视修改
#网络名称，同目录名称
Network="vit-base_RT2_ID4040_for_PyTorch"
#训练epoch
train_epochs=1
#训练batch_size
batch_size=64
#训练step
train_steps=`expr 1281167 / ${batch_size}`
#学习率
learning_rate=0.1
#二进制开关
bin_mode=False
bin_analysis=False

#维持参数，以下不需要修改
over_dump=False
data_dump_flag=False
data_dump_step="10"
profiling=False


if [[ $1 == --help || $1 == --h ]];then
   echo "usage:./train_performance_1p.sh --data_path=data_dir --batch_size=1024 --learning_rate=0.04"
   exit 1
fi

for para in $*
do
    if [[ $para == --data_path* ]];then
      data_path=`echo ${para#*=}`
    elif [[ $para == --ckpt_path* ]];then
      ckpt_path=`echo ${para#*=}`
    elif [[ $para == --precision_mode* ]];then
      precision_mode=`echo ${para#*=}`
    elif [[ $para == --batch_size* ]];then
      batch_size=`echo ${para#*=}`
    elif [[ $para == --learning_rate* ]];then
      learning_rate=`echo ${para#*=}`
    elif [[ $para == --bin_mode* ]];then
        bin_mode="True"
    elif [[ $para == --bin_analysis* ]];then
        bin_analysis="True"
    fi
done

PREC=""
if [[ $precision_mode == "amp" ]];then
  PREC="--apex"
fi

#校验是否传入data_path,不需要修改
if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
    exit 1
fi

#添加二进制代码
line=`grep "import torch" ${cur_path}/../train.py -n | tail -1|awk -F ':' '{print $1}'`
sed -i "$[line+1]itorch.npu.set_compile_mode(jit_compile=False)" ${cur_path}/../train.py

cd $cur_path
#设置环境变量，不需要修改
echo "Device ID: $ASCEND_DEVICE_ID"
export RANK_ID=$RANK_ID

if [ -d $cur_path/output ];then
   rm -rf $cur_path/output/*
   mkdir -p $cur_path/output/$ASCEND_DEVICE_ID
else
   mkdir -p $cur_path/output/$ASCEND_DEVICE_ID
fi
wait

#训练开始时间，不需要修改
start_time=$(date +%s)

nohup python3 $cur_path/../train.py  \
        --name cifar10-100_500 \
        --dataset cifar10 \
        --model_type ViT-B_16  \
        --pretrained_dir ${ckpt_path}/ViT-B_16.npz \
        --addr=127.0.0.1 \
        --train_batch_size=64 \
        --num_steps=781 \
        --eval_every=781 \
        --npu-fused-sgd \
        --fp16 \
        --data_dir ${data_path} \
        --fp16_opt_level O2 > $cur_path/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log 2>&1 &

wait

#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

sed -i "s|\r|\n|g"  $cur_path/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log
#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
FPS=`grep FPS $cur_path/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk -F "FPS=" '{print$2}' | awk -F ")" '{print$1}' |tail -n +5 |awk '{sum+=$1} END {print"",sum/NR}'|sed s/[[:space:]]//g`
#打印，不需要修改
echo "Final Performance images/sec : $FPS"

#输出训练精度,需要模型审视修改
train_accuracy=`grep -a "Best Accuracy:" $cur_path/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk '{print $3}'| awk '{print$NF}'`

#打印，不需要修改
echo "Final Train Accuracy : ${train_accuracy}"
echo "E2E Training Duration sec : $e2e_time"

#性能看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'perf'
#修改二进制用例名称
if [ $bin_mode == "True" ];then
    CaseName=$CaseName"_binary"
fi

#获取二进制支持算子
if [ $bin_analysis == "True" ];then
    cmd1=`ls -l /usr/local/Ascend/CANN-1.82/opp/op_impl/built-in/ai_core/tbe/kernel/config/ascend910|grep -v total|awk -F " " '{print $9}'|awk -F "." '{print $1}'`
    echo "cmd1=$cmd1" >> ${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log
fi

##获取性能数据，不需要修改
#吞吐量
ActualFPS=${FPS}
#单迭代训练时长
TrainingTime=`awk 'BEGIN{printf "%.2f\n", '${batch_size}'*1000/'${FPS}'}'`

#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
grep /781 $cur_path/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk -F "loss=" '{print$2}' | awk -F ")" '{print$1}' >> $cur_path/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt

#最后一个迭代loss值，不需要修改
ActualLoss=`awk 'END {print}' $cur_path/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt`

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
sed -i "s/ModuleNotFoundError: No module named "impl.lp_norm_reduce"/ /g" `grep ModuleNotFoundError -rl $cur_path/output/$ASCEND_DEVICE_ID/train_*.log`