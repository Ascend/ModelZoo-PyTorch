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
Network="SmartSketch_ID1046_for_PyTorch"
#训练epoch
train_epochs=0
#训练batch_size
batch_size=1
#训练step
train_steps=1
#学习率
learning_rate=8e-2
RANK_SIZE=1

#维测参数，precision_mode需要模型审视修改
#precision_mode="allow_mix_precision"
#维持参数，以下不需要修改
#over_dump=False
#data_dump_flag=False
#data_dump_step="10"
#profiling=False
PREC=""
# 帮助信息，不需要修改
if [[ $1 == --help || $1 == -h ]];then
    echo"usage:./train_performance_1p.sh <args>"
    echo " "
    echo "parameter explain:
    --precision_mode         precision mode(O0/O1/O2/O3)
    --data_path		             source data of training
    --train_steps                train steps
    -h/--help		                 show help message
    "
    exit 1
fi

#参数校验，不需要修改
for para in $*
do
    if [[ $para == --precision_mode* ]];then
        apex_opt_level=`echo ${para#*=}`
        if [[ $apex_opt_level != "O1" ]] && [[ $apex_opt_level != "O2" ]] && [[ $apex_opt_level != "O3" ]]; then
            echo "[Error] para \"precision_mode\" must be config O1 or O2 or O3"
            exit 1
        fi
        PREC="--apex --apex-opt-level "$apex_opt_level
    elif [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    elif [[ $para == --train_steps* ]];then
	train_steps=`echo ${para#*=}`
    fi
done

#校验是否传入data_path,不需要修改
if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
    exit 1
fi

mkdir -p /root/.cache/torch/checkpoints
cp /npu/traindata/ID1046_CarPeting_Pytorch_SmartSketch/vgg19-dcbb9e9d.pth /root/.cache/torch/checkpoints
#进入训练脚本目录，需要模型审视修改
cd $cur_path
#创建DeviceID输出目录，不需要修改
if [ -d ${cur_path}/test/output/${ASCEND_DEVICE_ID} ];then
   rm -rf ${cur_path}/test/output/${ASCEND_DEVICE_ID}
   mkdir -p ${cur_path}/test/output/$ASCEND_DEVICE_ID/ckpt
else
   mkdir -p ${cur_path}/test/output/$ASCEND_DEVICE_ID/ckpt
fi



#训练开始时间，不需要修改
start_time=$(date +%s)
nohup python3 ./backend/train.py --name=coco_pretrained \
                                    --dataset_mode=coco \
                                    --dataroot $data_path \
                                    --no_instance \
                                    --niter 2 \
                                    --max_steps 200 \
                                    --gpu_ids $ASCEND_DEVICE_ID \
                                    $PREC > $cur_path/test/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &
wait
#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))



#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
TrainingTime=`grep "time:" $cur_path/test/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk -F "time: " '{print $2}'|awk -F ")" '{print $1}'|awk '{sum+=$1} END {print sum/NR*1000}'`
FPS=`awk 'BEGIN{printf "%.2f\n", '${batch_size}'*1000/'${TrainingTime}'}'`
echo "Final Performance images/sec : $FPS"

#输出训练精度,需要模型审视修改
#train_accuracy=`grep "acc" $cur_path/test/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk -F ":" 'END {print $5}'|cut -c 1-6`#打印，不需要修改
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
#TrainingTime=`awk 'BEGIN{printf "%.2f\n", '${batch_size}'*1000/'${FPS}'}'`

#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
grep GAN $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk -F "GAN: " '{print $2}'|awk '{print $1}' > $cur_path/test/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt

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




