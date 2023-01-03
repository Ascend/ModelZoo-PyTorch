#!/bin/bash
cur_path=`pwd`/../

#Batch Size
batch_size=1
export RANKSIZE=1
export RANK_ID=0
RANK_ID_START=0
#网络名称，同目录名称
Network="CycleGAN_ID0521_for_PyTorch"
#Device数量，单卡默认为1
RankSize=1
#训练epoch，可选
train_epochs=1
#训练step
train_steps=5
#学习率
learning_rate=

#参数配置
data_path=""

if [[ $1 == --help || $1 == --h ]];then
        echo "usage:./train_performance_1p.sh "
        exit 1
fi

for para in $*
do
        if [[ $para == --data_path* ]];then
                data_path=`echo ${para#*=}`
        fi
done

if [[ $data_path  == "" ]];then
        echo "[Error] para \"data_path\" must be config"
        exit 1
fi

##############执行训练##########
cd $cur_path

mkdir -p ./outputs/$data_path
mkdir -p ./weights/$data_path/horse2zebra

if [ -d $cur_path/test/output ];then
        rm -rf $cur_path/test/output/*
        mkdir -p $cur_path/test/output/$ASCEND_DEVICE_ID
else
        mkdir -p $cur_path/test/output/$ASCEND_DEVICE_ID
fi
wait

start=$(date +%s)

nohup python3 train.py --dataset $data_path/horse2zebra --npu --epochs $train_epochs --batch-size $batch_size --max_steps $train_steps --decay_epochs 0 --apex > $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log 2>&1
wait

end=$(date +%s)
e2etime=$(( $end - $start ))

sed -i "s|\r|\n|g" $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log

#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修
FPS=`grep fps $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk -F 'fps: ' '{print $2}'|awk -F ':' '{print$1}'|tail -n +6|awk '{sum+=$1} END {print sum/NR}'`

#打印，不需要修改
#echo "Final Performance images/sec : $FPS"
#echo "Final Training Duration sec : $e2e_time"

#输出编译耗时
CompileTime=`grep "step_time" $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk -F "step_time" '{print $2}' | awk -F " " '{print $2}'| head -1 | awk '{sum+=$1} END {print"",sum}' |sed s/[[:space:]]//g`

#输出训练精度,需要模型审视修改
#train_accuracy=`grep "accuracy:" $cur_path/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk 'END {print $8}'`
#打印，不需要修改
#echo "Final Train Accuracy : ${train_accuracy}"

#性能看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RankSize}'p'_'perf'

##获取性能数据，不需要修改
#吞吐量
ActualFPS=${FPS}
#单迭代训练时长
TrainingTime=`awk 'BEGIN{printf "%.2f\n",'1000'*'${batch_size}'/'${FPS}'}'`

#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
grep Loss_D $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk -F 'Loss_D: ' '{print $2}'|awk '{print $1}'|awk '{if(length !=0) print $0}' > $cur_path/test/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt

#最后一个迭代loss值，不需要修改
ActualLoss=`awk 'END {print $1}' $cur_path/test/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt`

#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" > $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RankSize}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2etime}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CompileTime = ${CompileTime}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
rm -rf $data_path/horse2zebra/A
rm -rf $data_path/horse2zebra/B