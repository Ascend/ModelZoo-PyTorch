#!/bin/bash
currentDir=$(cd "$(dirname "$0")";pwd)/..

#当前路径,不需要修改
cur_path=`pwd`

#集合通信参数,不需要修改
export RANK_SIZE=8

# 数据集路径,保持为空,不需要修改
data_path=""

#网络名称,同目录名称,需要模型审视修改
Network="GAN"

#训练batch_size,,需要模型审视修改
batch_size=128



#参数校验，不需要修改
for para in $*
do
    if [[ $para == --device_id* ]];then
        device_id=`echo ${para#*=}`
    elif [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    fi
done

#校验是否传入data_path,不需要修改
if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
    exit 1
fi

# 校验是否指定了device_id,分动态分配device_id与手动指定device_id,此处不需要修改
if [ $ASCEND_DEVICE_ID ];then
    echo "device id is ${ASCEND_DEVICE_ID}"
elif [ ${device_id} ];then
    export ASCEND_DEVICE_ID=${device_id}
    echo "device id is ${ASCEND_DEVICE_ID}"
else
    export ASCEND_DEVICE_ID=0,1,2,3,4,5,6,7
    echo "device id is ${ASCEND_DEVICE_ID}"
fi

#训练开始时间，不需要修改
start_time=$(date +%s)
echo "start_time: ${start_time}"

#进入训练脚本目录，需要模型审视修改
cd $cur_path/

#创建DeviceID输出目录，不需要修改
if [ -d ${cur_path}/output/${Network}/${ASCEND_DEVICE_ID} ];then
    rm -rf ${cur_path}/output/${Network}/${ASCEND_DEVICE_ID}
    mkdir -p ${cur_path}/output/${Network}/$ASCEND_DEVICE_ID/ckpt
else
    mkdir -p ${cur_path}/output/${Network}/$ASCEND_DEVICE_ID/ckpt
fi

#非平台场景时source 环境变量
check_etp_flag=`env | grep etp_running_flag`
etp_flag=`echo ${check_etp_flag#*=}`
if [ x"${etp_flag}" != x"true" ];then
    source ${cur_path}/test/env_npu.sh
fi

#执行训练脚本，以下传参不需要修改，其他需要模型审视修改
export WORLD_SIZE=8
KERNEL_NUM=$(($(nproc)/8))
for((RANK_ID=0;RANK_ID<WORLD_SIZE;RANK_ID++));
do

    PID_START=$((KERNEL_NUM * RANK_ID))
    PID_END=$((PID_START + KERNEL_NUM - 1))
    taskset -c $PID_START-$PID_END python3 ./main.py \
        --distributed \
        --lr 0.0015 \
        --batch_size ${batch_size} \
        --n_epochs 200 \
        --workers 16 \
        --apex \
        --local_rank $RANK_ID \
        --data_path ${data_path} > ${cur_path}/output/train_full_8p.log 2>&1 &
done
wait

#训练结束时间，不需要修改
end_time=$(date +%s)
echo "end_time: ${end_time}"
e2e_time=$(( $end_time - $start_time ))

#最后一个迭代FPS值
FPS=`grep -a 'FPS:'  ${cur_path}/output/train_full_8p.log|awk -F "FPS:" '{print $NF}'|awk 'END {print}'`

#最后一个迭代loss值
D_loss=`grep -a 'D loss:'  ${cur_path}/output/train_full_8p.log | awk -F "D loss:" '{print $NF}'| awk 'END {print}' | awk -F "]" '{print $1}'`
G_loss=`grep -a 'G loss:'  ${cur_path}/output/train_full_8p.log | awk -F "G loss:" '{print $NF}'| awk 'END {print}' | awk -F "]" '{print $1}'`

#打印，不需要修改
echo "ActualFPS : $FPS"
echo "ActualLoss : D loss:${D_loss}, G loss:${G_loss}"
echo "E2E Training Duration sec : $e2e_time"

#稳定性精度看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'acc'

##获取性能数据，不需要修改
#单迭代训练时长
TrainingTime=`awk 'BEGIN{printf "%.2f\n", '${batch_size}'*1000/'${FPS}'}'`

#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要模型审视修改
grep -a 'D loss:'  ${cur_path}/output/train_full_8p.log | awk -F "D loss:" '{print $NF}' | awk -F "]" '{print $1}' >> $cur_path/output/${Network}/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt

grep -a 'FPS:'  ${cur_path}/output/train_full_8p.log | awk -F "FPS:" '{print $NF}' | awk -F "]" '{print $1}' >> $cur_path/output/${Network}/$ASCEND_DEVICE_ID/train_${CaseName}_FPS.txt

#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" > $cur_path/output/${Network}/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >> $cur_path/output/${Network}/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >> $cur_path/output/${Network}/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> $cur_path/output/${Network}/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> $cur_path/output/${Network}/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${FPS}" >> $cur_path/output/${Network}/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >> $cur_path/output/${Network}/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${loss}" >> $cur_path/output/${Network}/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> $cur_path/output/${Network}/$ASCEND_DEVICE_ID/${CaseName}.log






