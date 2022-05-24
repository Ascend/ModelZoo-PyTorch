#!/bin/bash

cur_path=`pwd`/../
path=`pwd`
#失败用例打屏

export HDF5_DISABLE_VERSION_CHECK=1

export PATH=/usr/local/hdf5/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/hdf5/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=/usr/local/hdf5/lib:$LIBRARY_PATH
export CPATH=/usr/local/hdf5/include:$CPATH

#基础参数，需要模型审视修改
#Batch Size
batch_size=160
#网络名称，同目录名称
Network="SRCNN_ID1770_for_PyTorch"
#Device数量，单卡默认为1
RankSize=8
#训练epoch，可选
train_epochs=10
#训练step
train_steps=
#学习率
learning_rate=1e-3

num_workers=192

#参数配置
data_path="/npu/traindata/ID1770_CarPeting_Pytorch_SRCNN"
PREC="--apex --apex_opt_level O2"

if [[ $1 == --help || $1 == --h ]];then
        echo "usage:./train_performance_1p.sh "
        exit 1
fi

for para in $*
do
        if [[ $para == --precision_mode* ]];then
        apex_opt_level=`echo ${para#*=}`
                if [[ $apex_opt_level != "O1" ]] && [[ $apex_opt_level != "O2" ]] && [[ $apex_opt_level != "O3" ]]; then
                        echo "[ERROR] para \"precision_mode\" must be config O1 or O2 or O3"
                        exit 1
                fi
        PREC="--apex --apex-opt-level "$apex_opt_level
        elif [[ $para == --data_path* ]];then
                data_path=`echo ${para#*=}`
        fi
done

if [[ $data_path  == "" ]];then
        echo "[Error] para \"data_path\" must be config"
        exit 1
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

export NPU_CALCULATE_DEVICE=$ASCEND_DEVICE_ID

start=$(date +%s)
export MASTER_ADDR=localhost
export MASTER_PORT=29688
export HCCL_WHITELIST_DISABLE=1

NPUS=($(seq 0 7))
export NPU_WORLD_SIZE=${#NPUS[@]}
rank=0
for i in ${NPUS[@]}
do
    mkdir -p  $path/output/${i}/
    export NPU_CALCULATE_DEVICE=${i}
    export RANK=${rank}
    export ASCEND_DEVICE_ID=${i}
    echo run process ${rank}
    python3 train.py $PREC --ddp --train-file "$data_path/SRCNN/91-image_x2.h5" \
                       --eval-file "$data_path/SRCNN/Set5_x2.h5" \
                       --outputs-dir "outputs" \
                       --scale 3 \
                       --lr $learning_rate \
                       --batch-size $batch_size \
                       --num-epochs $train_epochs  \
                       --num-workers $num_workers \
                       --seed 123 > $path/output/$ASCEND_DEVICE_ID/train_${i}.log 2>&1 &
    let rank++
done
wait
end=$(date +%s)
e2e_time=$(( $end - $start ))
sed -i "s|\r|\n|g" $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log
#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
FPS=`grep "FPS"  $path/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk -F ", loss" '{print$1}' | awk '{print$NF}' |awk '{sum+=$1} END {print"",sum*8/NR}'`
#打印，不需要修改
echo "Final Performance images/sec : $FPS"

#打印，不需要修改
echo "E2E Training Duration sec : $e2e_time"

#性能看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RankSize}'p'_'perf'

#获取性能数据，不需要修改
ActualFPS=${FPS}

#单迭代训练时长
TrainingTime=`echo "${BatchSize} ${FPS}"|awk '{printf("%.4f\n", $1*1000/$2)}'`

#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
grep "FPS"  $path/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log| awk '{print$NF}' > $path/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt

#最后一个迭代loss值，不需要修改
ActualLoss=`awk 'END {print}' $path/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt`

#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" > $path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RankSize}" >> $path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >> $path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> $path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> $path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >> $path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >> $path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> $path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> $path/output/$ASCEND_DEVICE_ID/${CaseName}.log