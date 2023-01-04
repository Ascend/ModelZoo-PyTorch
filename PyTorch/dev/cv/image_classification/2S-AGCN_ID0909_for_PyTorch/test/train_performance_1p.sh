#!/bin/bash

#当前路径,不需要修改
cur_path=`pwd`/../

#集合通信参数,不需要修改


export RANK_SIZE=1
export JOB_ID=10087
RANK_ID_START=0

# 数据集路径,保持为空,不需要修改
data_path=""

#设置默认日志级别,不需要修改
#export ASCEND_GLOBAL_LOG_LEVEL=3
#export ASCEND_SLOG_PRINT_TO_STDOUT=0

#基础参数，需要模型审视修改
#网络名称，同目录名称
Network="2S-AGCN_ID0909_for_PyTorch"
#训练epoch
train_epochs=0
#训练batch_size
batch_size=32
#训练step
train_steps=1
#学习率
learning_rate=8e-2
RANK_SIZE=1

time_limit=0

#维测参数，precision_mode需要模型审视修改
#precision_mode="allow_mix_precision"
#维持参数，以下不需要修改
#over_dump=False
#data_dump_flag=False
#data_dump_step="10"
#profiling=False

# 帮助信息，不需要修改
if [[ $1 == --help || $1 == -h ]];then
    echo"usage:./train_performance_1p.sh <args>"
    echo " "
    echo "parameter explain:
    --data_path		           source data of training
    --train_steps                  train steps
    -h/--help		             show help message
    "
    exit 1
fi

#参数校验，不需要修改
for para in $*
do
    if [[ $para == --precision_mode* ]];then
        precision_mode=`echo ${para#*=}`
    elif [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    elif [[ $para == --time_limit* ]];then
        time_limit=`echo ${para#*=}`
    elif [[ $para == --train_steps* ]];then
	train_steps=`echo ${para#*=}`
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
cd $cur_path
#创建DeviceID输出目录，不需要修改
if [ -d ${cur_path}/test/output/${ASCEND_DEVICE_ID} ];then
   rm -rf ${cur_path}/test/output/${ASCEND_DEVICE_ID}
   mkdir -p ${cur_path}/test/output/$ASCEND_DEVICE_ID/ckpt
else
   mkdir -p ${cur_path}/test/output/$ASCEND_DEVICE_ID/ckpt
fi

sed -i "s|./data/ntu/xview/train_data_joint.npy|$data_path/ntu/xview/train_data_joint.npy|g" config/nturgbd-cross-view/train_joint.yaml
sed -i "s|./data/ntu/xview/train_label.pkl|$data_path/ntu/xview/train_label.pkl|g" config/nturgbd-cross-view/train_joint.yaml
sed -i "s|./data/ntu/xview/val_data_joint.npy|$data_path/ntu/xview/val_data_joint.npy|g" config/nturgbd-cross-view/train_joint.yaml
sed -i "s|./data/ntu/xview/val_label.pkl|$data_path/ntu/xview/val_label.pkl|g" config/nturgbd-cross-view/train_joint.yaml


#训练开始时间，不需要修改
start_time=$(date +%s)
#rm -rf ./runs/ntu_cv_agcn_joint
nohup python3 main.py --config ./config/nturgbd-cross-view/train_joint.yaml \
                                    --batch-size 32 \
                                    --step 20 \
                                    --num-epoch 1 \
                                    --device $ASCEND_DEVICE_ID \
                                    --time_limit ${time_limit} \
                                    $PREC > $cur_path/test/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &
wait
#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))



#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
FPS=`grep "FPS" $cur_path/test/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk -F ":" 'END {print $2}'|cut -c 3-7 `
echo "Final Performance images/sec : $FPS"
#输出编译耗时
CompileTime=`grep "Time:" $cur_path/test/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | head -2 |awk -F ":" '{print $2}' | awk '{sum+=$1} END {print"",sum}' |sed s/[[:space:]]//g`


#输出训练精度,需要模型审视修改
train_accuracy=`grep "best accuracy" $cur_path/test/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk -F ":" 'END {print $2}'|cut -c 2-6`
echo "Final Train Accuracy : ${train_accuracy}"
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
TrainingTime=`awk 'BEGIN{printf "%.2f\n", '${batch_size}'*1000/'${FPS}'}'`

#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
grep "Mean training loss" $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk -F ":" 'END {print $2}'|cut -c 2-4|sed 's/,//g'|sed '/^$/d' >> $cur_path/test/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt

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
###下面字段用于冒烟看护
#BatchSize=${batch_size}
#设备类型，自动获取
#DeviceType=`uname -m`
#用例名称，自动获取
#CaseName=${Network}_bs${BatchSize}_${RankSize}'p'_'perf'


