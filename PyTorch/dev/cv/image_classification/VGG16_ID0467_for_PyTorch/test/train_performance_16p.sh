#!/bin/bash

#当前路径,不需要修改
cur_path=`pwd`

#集合通信参数,不需要修改

export RANK_SIZE=16
export JOB_ID=10087
RANK_ID_START=0

# 数据集路径,保持为空,不需要修改
data_path=""
conf_path=""
server_index=""
fix_node_ip=""
devicesnum=""

#基础参数，需要模型审视修改
#网络名称，同目录名称
Network="VGG16_ID0467_for_PyTorch"
#训练epoch
train_epochs=1
#训练batch_size
batch_size=32
#训练step
train_steps=
#学习率
learning_rate=0.016
#维持参数，审视修改
precision_mode="amp"
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
    elif [[ $para == --precision_mode* ]];then
        precision_mode=`echo ${para#*=}`
    elif [[ $para == --batch_size* ]];then
      batch_size=`echo ${para#*=}`
    elif [[ $para == --learning_rate* ]];then
      learning_rate=`echo ${para#*=}`
    elif [[ $para == --epochs* ]];then
      train_epochs=`echo ${para#*=}`
    elif [[ $para == --fix_node_ip* ]];then
	    fix_node_ip=`echo ${para#*=}`
	elif [[ $para == --devicesnum* ]];then
	    devicesnum=`echo ${para#*=}`
    elif [[ $para == --conf_path* ]];then
        conf_path=`echo ${para#*=}`
    elif [[ $para == --server_index* ]];then
        server_index=`echo ${para#*=}`
    fi
done

one_node_ip=`find $conf_path -name "server_*0.info"|awk -F "server_" '{print $2}'|awk -F "_" '{print $1}'`
linux_num=`find $conf_path -name "server_*.info" |wc -l`

PREC=""
if [[ $precision_mode == "amp" ]];then
  PREC="--apex"
fi

#校验是否传入data_path,不需要修改
if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
    exit 1
fi

#################创建日志输出目录，不需要修改#################
if [ -d $cur_path//output/${ASCEND_DEVICE_ID} ];then
    rm -rf $cur_path/output/${ASCEND_DEVICE_ID}
    mkdir -p $cur_path/output/$ASCEND_DEVICE_ID
else
    mkdir -p $cur_path/output/$ASCEND_DEVICE_ID
fi


#训练开始时间，不需要修改
cd $cur_path
start_time=$(date +%s)

export HCCL_IF_IP=$fix_node_ip
export MASTER_ADDR=$one_node_ip
export MASTER_PORT=29688
export HCCL_WHITELIST_DISABLE=1
device_num=${#devicesnum}
devices_num=`awk 'BEGIN{printf "%.0f\n",'${device_num}'-1}'`

rank_server=`awk 'BEGIN{printf "%.0f\n",'${device_num}'*'${server_index}'}'`
export WORLD_SIZE=`awk 'BEGIN{printf "%.0f\n",'${device_num}'*'${linux_num}'}'`

NPUS=($(seq 0 7))
rank=0
for i in ${NPUS[@]}
do 
    export NPU_CALCULATE_DEVICE=${i}
	export RANK=`awk 'BEGIN{printf "%.0f\n",'${rank_server}'+'${rank}'}'`
    export LOCAL_RANK=${rank}
	echo run process ${rank}

    nohup python3.7 $cur_path/../train.py  \
        --model vgg16 \
        --epochs ${train_epochs} \
        --device ${rank} \
        --max_steps=2000 \
        --data-path=${data_path} \
        --batch-size ${batch_size} $PREC \
        --lr ${learning_rate} \
        --momentum 0.9 \
        --weight-decay 1e-4 \
        --print-freq 10 \
        --dist-url='tcp://127.0.0.1:50000' > $cur_path/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log 2>&1 &
    let rank++
done
wait

#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
FPS=`grep Epoch: $cur_path/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk -F "img/s:" '{print$2}' | awk '{print$1}'|tail -n +5|awk '{sum+=$1} END {print sum/NR}'`
#打印，不需要修改
echo "fps : $fps"
echo "Final Performance images/sec : $FPS"

#输出训练精度,需要模型审视修改
train_accuracy=`grep -a '* Acc@1' $cur_path/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk '{print $3}'| awk 'BEGIN {max = 0} {if ($1+0>max+0) max=$1 fi} END {print max}'`

#打印，不需要修改
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
grep Epoch: $cur_path/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log| awk -F "loss:" '{print$2}' | awk '{print$1}' > $cur_path/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt

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
