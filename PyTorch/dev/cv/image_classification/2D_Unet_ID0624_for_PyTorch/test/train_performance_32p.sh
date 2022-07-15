#!/bin/bash

cur_path=`pwd`/../
#失败用例打屏
export ASCEND_SLOG_PRINT_TO_STDOUT=0

source /usr/local/Ascend/bin/setenv.bash

export PATH=/usr/local/hdf5/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/hdf5/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=/usr/local/hdf5/lib:$LIBRARY_PATH
export CPATH=/usr/local/hdf5/include:$CPATH

#基础参数，需要模型审视修改
#Batch Size
batch_size=1
#网络名称，同目录名称
Network="2D_Unet_ID0624_for_PyTorch"
#Device数量，单卡默认为1
RANK_SIZE=32
#训练epoch，可选
train_epochs=1
#训练step
train_steps=
#学习率
learning_rate=1e-3
#参数配置
data_path=""
conf_path=""
server_index=""
fix_node_ip=""

if [[ $1 == --help || $1 == --h ]];then
        echo "usage:./train_performance_1p.sh "
        exit 1
fi

for para in $*
do
        if [[ $para == --data_path* ]];then
                data_path=`echo ${para#*=}`
        elif [[ $para == --conf_path* ]];then
          conf_path=`echo ${para#*=}`
  elif [[ $para == --server_index* ]];then
          server_index=`echo ${para#*=}`
  elif [[ $para == --fix_node_ip* ]];then
          fix_node_ip=`echo ${para#*=}`
        fi
done

if [[ $data_path  == "" ]];then
        echo "[Error] para \"data_path\" must be config"
        exit 1
fi

one_node_ip=`find $conf_path -name "server_*0.info"|awk -F "server_" '{print $2}'|awk -F "_" '{print $1}'`
linux_num=`find $conf_path -name "server_*.info" |wc -l`

export HCCL_IF_IP=$fix_node_ip
export MASTER_ADDR=$one_node_ip

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
sed -i "s|data/imgs/|$data_path/imgs/|g" $cur_path/train.py
sed -i "s|data/masks/|$data_path/masks/|g" $cur_path/train.py
#sed -i "s|if global_step == 100: pass|if global_step == 100: break|g" $cur_path/train.py
start=$(date +%s)
#nohup python3 train.py -e $train_epochs > $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log 2>&1 &


export HCCL_WHITELIST_DISABLE=1
export MASTER_PORT=23456
NPUS=($(seq 0 7))
rank_server=`awk 'BEGIN{printf "%.0f\n",8*'${server_index}'}'`
export NPU_WORLD_SIZE=`awk 'BEGIN{printf "%.0f\n",8*'${linux_num}'}'`
rank=0
for i in ${NPUS[@]}
do
    mkdir -p  $cur_path/test/output/${i}/
    export NPU_CALCULATE_DEVICE=${i}
    export ASCEND_DEVICE_ID=${i}
    export RANK=`awk 'BEGIN{printf "%.0f\n",'${rank}'+'${rank_server}'}'`
    echo run process ${rank}
    nohup python3 train.py -e $train_epochs  --distributed True > $cur_path/test/output/$ASCEND_DEVICE_ID/train_${i}.log 2>&1 &
    let rank++
done
wait
end=$(date +%s)
e2e_time=$(( $end - $start ))


#sed -i "s|if global_step == 100: break|if global_step == 100: pass|g" $cur_path/train.py

#输出训练精度,需要模型审视修改
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
sed -i "s|\r|\n|g" $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log
TrainingTime=0
FPS=`grep img/s $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log | grep -v 0% | awk -F "," '{print$2}' | awk '{print$1}' | awk -F "i" '{print$1}' | awk '{sum+=$1} END {print"",sum/NR}'|sed s/[[:space:]]//g`
FPS=$(awk 'BEGIN{print '$FPS'*32}')

TrainingTime=`awk 'BEGIN{printf "%.2f\n",'${batch_size}'*1000/'${FPS}'}'`

#打印，不需要修改
echo "Final Performance images/sec : $FPS"

ActualFPS=${FPS}

#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
grep Epoch $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log| awk -F "," '{print$3}' | awk -F "=" '{print$2}' | awk -F "]" '{print$1}'| awk '{if(length !=0)print $0}' > $cur_path/test/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt


#精度值
#train_accuracy=`grep "loss" $cur_path/test/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss_2.txt|awk -F " " '{print $8}'|awk 'END {print}'`

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
#echo "TrainAccuracy = ${train_accuracy}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
