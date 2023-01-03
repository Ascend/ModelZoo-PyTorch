#!/bin/bash

#当前路径,不需要修改
cur_path=`pwd`/../

#集合通信参数,不需要修改
export RANK_SIZE=1
export JOB_ID=10087
RANK_ID_START=0
export NPU_CALCULATE_DEVICE=$ASCEND_DEVICE_ID

# 数据集路径,保持为空,不需要修改
data_path=""

#设置默认日志级别,不需要修改
#export ASCEND_GLOBAL_LOG_LEVEL=3
#export ASCEND_SLOG_PRINT_TO_STDOUT=0

#基础参数，需要模型审视修改
#网络名称，同目录名称
Network="SRGAN_ID2956_for_PyTorch"
#训练epoch
train_epochs=1
#训练batch_size
batch_size=16
#训练step
train_steps=
#学习率
learning_rate=
#打印频率
print_freq=1

#维测参数，precision_mode需要模型审视修改
precision_mode="allow_mix_precision"
#维持参数，以下不需要修改
over_dump=False
data_dump_flag=False
data_dump_step="10"
profiling=False
autotune=False
PREC=""
# 帮助信息，不需要修改

if [[ $1 == --help || $1 == -h ]];then
    echo"usage:./train_performance_1p.sh <args>"
    echo " "
    echo "parameter explain:
    --precision_mode         precision mode(allow_fp32_to_fp16/force_fp16/must_keep_origin_dtype/allow_mix_precision)
    --over_dump                  if or not over detection, default is False
    --data_dump_flag         data dump flag, default is False
    --data_dump_step             data dump step, default is 10
    --profiling                  if or not profiling for performance debug, default is False
    --data_path                  source data of training
    -h/--help                    show help message
    "
    exit 1
fi

#参数校验，不需要修改
for para in $*
do
    if [[ $para == --precision_mode* ]];then
        apex_opt_level=`echo ${para#*=}`
                    if [[ $apex_opt_level != "O1" ]] && [[ $apex_opt_level != "O2" ]] && [[ $apex_opt_level != "O3" ]]; then
                            echo "[ERROR] para \"precision_mode\" must be config O1 or O2 or O3"
                            exit 1
                    fi
        PREC="--apex --apex-opt-level "$apex_opt_level

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


#校验是否传入data_path,不需要修改
if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
    exit 1
fi

#进入训练脚本目录，需要模型审视修改
cd $cur_path
#创建DeviceID输出目录，不需要修改
if [ -d ${cur_path}/test/output/${ASCEND_DEVICE_ID} ];then
   rm -rf ${cur_path}/test/output/${ASCEND_DEVICE_ID}
   mkdir -p ${cur_path}/test/output/$ASCEND_DEVICE_ID/ckpt
else
   mkdir -p ${cur_path}/test/output/$ASCEND_DEVICE_ID/ckpt
fi


#参数修改 
sed -i "s|'/media/ssd/sr data/train2014'|'$data_path/coco2014/train2014'|g" create_data_lists.py
sed -i "s|'/media/ssd/sr data/BSDS100'|'$data_path/coco2014/val2014'|g" create_data_lists.py
sed -i "s|srresnet_checkpoint = \"./checkpoint_srresnet.pth.tar\"|srresnet_checkpoint = \"$data_path/checkpoint_srresnet.pth.tar\"|g" train_srgan.py
sed -i "s|batch_size = 16|batch_size = $batch_size|g" train_srgan.py
sed -i "s|print_freq = 500|print_freq = $print_freq|g" train_srgan.py
sed -i "s|for epoch in range(start_epoch, epochs)|for epoch in range(start_epoch, $train_epochs)|g" train_srgan.py
sed -i "s|pass|break|g" train_srgan.py
 
#训练开始时间，不需要修改
start_time=$(date +%s)

nohup python3 create_data_lists.py > $cur_path/test/output/${ASCEND_DEVICE_ID}/dataset_prepare.log 2>&1 &
wait
nohup python3 train_srgan.py > $cur_path/test/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &

wait
#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

#参数回改 
sed -i "s|'$data_path/coco2014/train2014'|'/media/ssd/sr data/train2014'|g" create_data_lists.py
sed -i "s|'$data_path/coco2014/val2014'|'/media/ssd/sr data/BSDS100'|g" create_data_lists.py
sed -i "s|srresnet_checkpoint = \"$data_path/checkpoint_srresnet.pth.tar\"|srresnet_checkpoint = \"./checkpoint_srresnet.pth.tar\"|g" train_srgan.py
sed -i "s|batch_size = $batch_size|batch_size = 16|g" train_srgan.py
sed -i "s|print_freq = $print_freq|print_freq = 500|g" train_srgan.py
sed -i "s|for epoch in range(start_epoch, $train_epochs)|for epoch in range(start_epoch, epochs)|g" train_srgan.py
sed -i "s|break|pass|g" train_srgan.py



sed -i "s|\r|\n|g" $cur_path/test/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log
 
#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
FPS=`grep FPS $cur_path/test/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk -F 'FPS ' '{print $2}'|tail -n +5|awk '{sum+=$1} END {print"",sum/NR}'|sed s/[[:space:]]//g`
#打印，不需要修改
echo "Final Performance images/sec : $FPS"

#获取编译时间
CompileTime=`grep "Epoch:" $cur_path/test/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | head -2 | awk -F " " '{print $4}' | awk '{sum+=$1} END {print"",sum}' |sed s/[[:space:]]//g`

#输出训练精度,需要模型审视修改
#train_accuracy="NULL"
#打印，不需要修改
#echo "Final Train Accuracy : ${train_accuracy}"
echo "E2E Training Duration sec : $e2e_time"

#性能看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'perf'

##获取性能数据，不需要修改
#吞吐量
ActualFPS=${FPS}
#单迭代训练时长
TrainingTime=`awk 'BEGIN{printf "%.2f\n",'${batch_size}'*1000/'${ActualFPS}'}'`

#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
grep FPS $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk -F 'Cont. Loss ' '{print $2}'|awk '{print $1}' > $cur_path/test/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt

#最后一个迭代loss值，不需要修改
ActualLoss=`awk 'END {print}' $cur_path/test/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt`

#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" > $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CompileTime = ${CompileTime}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
#echo "TrainAccuracy = ${train_accuracy}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
