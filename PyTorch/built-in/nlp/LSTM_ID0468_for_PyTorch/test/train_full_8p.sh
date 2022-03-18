#!/bin/bash

#当前路径,不需要修改
cur_path=`pwd`

#集合通信参数,不需要修改

export RANK_SIZE=8
export JOB_ID=10087
RANK_ID_START=0

# 数据集路径,保持为空,不需要修改
data_path=""

#基础参数，需要模型审视修改
#网络名称，同目录名称
Network="LSTM_ID0468_for_PyTorch"
#训练epoch
train_epochs=30
#训练batch_size
batch_size=1024
#训练step
train_steps=
#学习率
learning_rate=0.001

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
    --over_dump		           if or not over detection, default is False
    --data_dump_flag		     data dump flag, default is False
    --data_dump_step		     data dump step, default is 10
    --profiling		           if or not profiling for performance debug, default is False
	--data_path		           source data of training
    -h/--help		             show help message
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

#进入训练脚本目录，需要模型审视修改
cd $cur_path/../NPU/8p/

#训练前修改参数配置
sed -i "s|data|${data_path}|g" conf/ctc_config.yaml
sed -i "s|from torch.utils.tensorboard import SummaryWriter|from tensorboardX import SummaryWriter|g" steps/train_ctc.py
sed -i "s|data|${data_path}|g" ${data_path}/train/fbank.scp
sed -i "s|data|${data_path}|g" ${data_path}/dev/fbank.scp
sed -i "s|data|${data_path}|g" ${data_path}/test/fbank.scp

#训练开始时间，不需要修改
start_time=$(date +%s)

#设置环境变量，不需要修改
echo "Device ID: $ASCEND_DEVICE_ID"

#创建DeviceID输出目录，不需要修改
if [ -d ${cur_path}/output/${ASCEND_DEVICE_ID} ];then
	rm -rf ${cur_path}/output/${ASCEND_DEVICE_ID}
	mkdir -p ${cur_path}/output/$ASCEND_DEVICE_ID/ckpt
else
	mkdir -p ${cur_path}/output/$ASCEND_DEVICE_ID/ckpt
fi

#执行训练脚本，以下传参不需要修改，其他需要模型审视修改
#--data_dir, --model_dir, --precision_mode, --over_dump, --over_dump_path，--data_dump_flag，--data_dump_step，--data_dump_path，--profiling，--profiling_dump_path
nohup python3 -u steps/train_ctc.py \
--rank 0 \
--world_size 1 \
--dist_backend 'hccl' \
--dist_url 'tcp://127.0.0.1:50000' \
--multiprocessing_distributed \
--device_list 0,1,2,3,4,5,6,7 \
--device_id $ASCEND_DEVICE_ID \
$PREC \
--loss_scale 128 \
--opt_level O2 \
--conf 'conf/ctc_config.yaml' \
--addr $(hostname -I |awk '{print $1}') > ${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &
wait

#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

#训练完恢复参数配置
sed -i "s|${data_path}|data|g" conf/ctc_config.yaml
sed -i "s|from tensorboardX import SummaryWriter|from torch.utils.tensorboard import SummaryWriter|g"  steps/train_ctc.py
sed -i "s|${data_path}|data|g" ${data_path}/train/fbank.scp
sed -i "s|${data_path}|data|g" ${data_path}/dev/fbank.scp
sed -i "s|${data_path}|data|g" ${data_path}/test/fbank.scp

#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
FPS=`grep "Epoch:" $cur_path/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk 'END {print $11}'|sed 's/,$//'`
#打印，不需要修改
echo "Final Performance item/sec : $FPS"

#输出训练精度,需要模型审视修改
train_accuracy=`grep "acc is:" ${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk 'END {print $10}'`
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
TrainingTime=`awk 'BEGIN{printf "%.2f\n",'${BatchSize}'*1000/'${FPS}'}'`

#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
grep "Epoch:" $cur_path/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk '{print $14}'|sed 's/,$//' >> $cur_path/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt

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
echo "TrainAccuracy = ${train_accuracy}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log