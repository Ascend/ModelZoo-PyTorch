#!/bin/bash

export NPU_CALCULATE_DEVICE=$ASCEND_DEVICE_ID
cur_path=`pwd`/../
export RANK_SIZE=1
export JOB_ID=10087
RANK_ID_START=0

#基础参数，需要模型审视修改
batch_size=256
#网络名称，同目录名称
Network="TransformerXL_RT2_ID4047_for_PyTorch"
#Device数量，单卡默认为1
RankSize=1
#训练epoch，可选
train_epochs=
#训练step
train_steps=40000
#学习率
learning_rate=0.495

#参数配置
data_path=""
#维测参数，precision_mode需要模型审视修改
precision_mode="allow_mix_precision"

if [[ $1 == --help || $1 == --h ]];then
    echo "usage:./train_performance_1p.sh "
    exit 1
fi

for para in $*
do
    if [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    elif [[ $para == --precision_mode* ]];then
        precision_mode=`echo ${para#*=}`
    elif [[ $para == --fp32* ]];then
        fp32=`echo ${para#*=}`
    elif [[ $para == --hf32* ]];then
        hf32=`echo ${para#*=}`
    fi
done

if [[ $data_path  == "" ]];then
    echo "[Error] para \"data_path\" must be config"
    exit 1
fi

if [[ $precision_mode == "must_keep_origin_dtype" ]];then
   prec=""
else
   prec="--fp16"
fi

##############执行训练##########
cd $cur_path
if [ -d $cur_path/test/output ];then
    rm -rf $cur_path/test/output/*
    mkdir -p $cur_path/test/output/$ASCEND_DEVICE_ID
else
    mkdir -p $cur_path/test/output/$ASCEND_DEVICE_ID
fi
wait


start=$(date +%s)
nohup python3 -m torch.distributed.launch \
    --nproc_per_node=1 \
    $cur_path/pytorch/train.py \
    --affinity='disabled' \
    --config_file pytorch/wt103_base.yaml \
    --config aiserver_1npu_fp32 \
    --work_dir=$cur_path/test/output/$ASCEND_DEVICE_ID \
    --batch_size=$batch_size \
    --batch_chunk=16 \
    --data=$data_path \
    --max_step=$train_steps \
    --precision_mode=$precision_mode ${fp32} ${hf32} ${prec} > $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log 2>&1 &
wait
end=$(date +%s)
e2e_time=$(( $end - $start ))

###下面字段用于冒烟看护
BatchSize=${batch_size}
#设备类型，自动获取
DeviceType=`uname -m`
#用例名称，自动获取
if [[ ${fp32} == "--fp32" ]];then
  CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'fp32'_'acc'
elif [[ ${hf32} == "--hf32" ]];then
  CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'hf32'_'acc'
else
  CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'acc'
fi
#修改二进制用例名称
if [ $bin_mode == "True" ];then
    CaseName=$CaseName"_binary"
fi

#结果打印，不需要修改
echo "-------------------- Final result --------------------"
#输出性能FPS，需要模型审视修改
FPS=`grep "Training throughput" $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log |awk '{print$3}'`
ActualFPS=${FPS}
#打印，不需要修改
echo "Final performanceimages/sec: $ActualFPS"
echo "Final Training Duration sec : $e2e_time"
#单迭代训练时长
TrainingTime=`awk 'BEGIN{printf "%.2f\n",'${BatchSize}'*1000/'${FPS}'}'`

#输出训练精度,需要模型审视修改
train_accuracy=`grep Eval $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log| awk '{print$17}' | awk 'NR==1{min=$1;next}{min=min<$1?min:$1}END{print min}'`

#从train_$ASCEND_DEVICE_ID.log提取loss到train_${CaseName}_loss.txt中，需要根据模型审视
grep "ms/batch" $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|grep -v "train loss"|awk -F 'loss ' '{print $2}'|awk -F '|' '{print $1}' > $cur_path/test/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt
#最后一个迭代loss值，不需要修改
ActualLoss=`awk 'END {print $1}' $cur_path/test/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt`

#关键信息打印到CaseName.log中，此处无需修改
echo "Network = ${Network}" > $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RankSize}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainAccuracy = ${train_accuracy}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log