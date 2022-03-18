#!/bin/bash

#当前路径,不需要修改
cur_path=`pwd`

#规避环境变量冲突
if [ -f /usr/local/Ascend/bin/setenv.bash ];then
    unset PYTHONPATH
    source /usr/local/Ascend/bin/setenv.bash  
fi
#集合通信参数,不需要修改
export RANK_SIZE=8
RANK_ID_START=0

# 数据集路径,保持为空,不需要修改
data_path=""
# 训练周期
epochs=1
#网络名称,同目录名称,需要模型审视修改
Network="DenseNet161_ID0455_for_Pytorch"

#训练batch_size,,需要模型审视修改
batch_size=1024
#训练steps
train_steps=10

#参数校验，不需要修改
for para in $*
do
    if [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    elif [[ $para == --epochs* ]];then
        epochs=`echo ${para#*=}`
    fi
done

#校验是否传入data_path,不需要修改
if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
    exit 1
fi

#训练开始时间，不需要修改
start_time=$(date +%s)

#进入训练脚本目录，需要模型审视修改
cd $cur_path/../references/classification

#修改参数
sed -i "s|pass|break|g"  $cur_path/../references/classification/utils.py
wait
for((RANK_ID=$RANK_ID_START;RANK_ID<$((RANK_SIZE+RANK_ID_START));RANK_ID++));
do
    #设置环境变量，不需要修改
    echo "Device ID: $RANK_ID"
    export RANK_ID=$RANK_ID
    export ASCEND_DEVICE_ID=$RANK_ID
    ASCEND_DEVICE_ID=$RANK_ID

    #创建DeviceID输出目录，不需要修改
    if [ -d ${cur_path}/output/${ASCEND_DEVICE_ID} ];then
        rm -rf ${cur_path}/output/${ASCEND_DEVICE_ID}
        mkdir -p ${cur_path}/output/$ASCEND_DEVICE_ID/ckpt
    else
        mkdir -p ${cur_path}/output/$ASCEND_DEVICE_ID/ckpt
    fi

    # 非平台场景时source 环境变量
    check_etp_flag=`env | grep etp_running_flag`
    etp_flag=`echo ${check_etp_flag#*=}`
    if [ x"${etp_flag}" != x"true" ];then
        source ${cur_path}/env_npu.sh
    fi

    # 绑核，不需要的绑核的模型删除，需要的模型审视修改
    #let a=RANK_ID*12
    #let b=RANK_ID+1
    #let c=b*12-1

    #执行训练脚本，以下传参不需要修改，其他需要模型审视修改
    python3 train.py  \
        --model densenet161 \
        --epochs ${epochs} \
        --max_steps ${train_steps} \
        --data-path=$data_path \
        --distributed \
        --batch-size=$batch_size \
        --workers 128 \
        --lr 0.8 \
        --momentum 0.9 \
        --weight-decay 1e-4 \
        --apex \
        --apex-opt-level O2 \
        --loss_scale_value 1024 \
        --seed 1234 \
        --print-freq 1 > ${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &
done
wait

#8p情况下仅0卡(主节点)有完整日志,因此后续日志提取仅涉及0卡
ASCEND_DEVICE_ID=0

#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

#参数回改
sed -i "s|break|pass|g"  $cur_path/../references/classification/utils.py
wait

#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
FPS=`grep -a 'Epoch:'  $cur_path/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|grep eta:|awk -F "img/s: " '{print $NF}'|awk 'NR==1{max=$1;next}{max=max>$1?max:$1}END{print max}'`

#打印，不需要修改
echo "E2E Training Duration sec : $e2e_time"

#性能看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'perf'

##获取性能数据，不需要修改
#吞吐量
ActualFPS=`awk -v x="$FPS" -v y="$RANK_SIZE" 'BEGIN{printf "%.3f\n", x*y}'`
#单迭代训练时长
TrainingTime=`awk 'BEGIN{printf "%.2f\n", '${batch_size}'*1000/'${FPS}'}'`

#打印，不需要修改
echo "Final Performance images/sec : $ActualFPS"

#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要模型审视修改
grep Epoch: $cur_path/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|grep eta:|awk -F "loss: " '{print $NF}' | awk -F " " '{print $1}' > $cur_path/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt

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