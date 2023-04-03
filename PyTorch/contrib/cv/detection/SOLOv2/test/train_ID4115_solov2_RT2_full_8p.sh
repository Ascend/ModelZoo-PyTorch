#!/bin/bash

#当前路径,不需要修改
cur_path=`pwd`

#集合通信参数
export RANK_SIZE=8
#设置log level,0-debug/1-info/2-warning/3-error
#export ASCEND_GLOBAL_LOG_LEVEL_ETP=3
#将Host日志输出到串口,0-关闭/1-开启
#export ASCEND_SLOG_PRINT_TO_STDOUT=0
#export PYTHONPATH=${cur_path}/../mmcv:$PYTHONPATH
# 数据集路径,保持为空,不需要修改
data_path=""
apex="O1"
#网络名称,同目录名称,需要模型审视修改
Network="solov2_RT2_ID4115_for_Pytorch"

epochs=12
#训练batch_size,,需要模型审视修改
batch_size=16
#npu 设备ID
device_id=4
#FPS输出间隔
fps_lag=200
#profiling输出的起始点
start_step=0
#profiling输出的终止点
stop_step=20

#输出profiling的类型，可以选择：None、CANN、GE
profiling=None
#是否使能二进制编译，0->否，1->是
rt2_bin=1
#conda环境的名称
conda_name=py1
#loss输出的间隔
interval=50

#参数校验，不需要修改
for para in $*
do
    if [[ $para == --device_id* ]];then
        device_id=`echo ${para#*=}`
    elif [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    elif [[ $para == --apex* ]];then
        apex=`echo ${para#*=}`
    elif [[ $para == --fps_lag* ]];then
        fps_lag=`echo ${para#*=}`
    elif [[ $para == --start_step* ]];then
        start_step=`echo ${para#*=}`
    elif [[ $para == --stop_step* ]];then
        stop_step=`echo ${para#*=}`
    elif [[ $para == --steps_per_epoch* ]];then
        steps_per_epoch=`echo ${para#*=}`
    elif [[ $para == --profiling* ]];then
        profiling=`echo ${para#*=}`
    elif [[ $para == --rt2_bin* ]];then
        rt2_bin=`echo ${para#*=}`
    elif [[ $para == --conda_name* ]];then
        conda_name=`echo ${para#*=}`
        source set_conda.sh
        source activate $conda_name
    fi
done

if [[ $profiling == "GE" ]];then
    export GE_PROFILING_TO_STD_OUT=1
    echo "GEEEEEEEE!"
fi

#校验是否传入data_path,不需要修改
if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
    exit 1
fi

# 校验是否指定了device_id,分动态分配device_id与手动指定device_id,此处不需要修改

if [ $ASCEND_DEVICE_ID ];then
    device_id=$ASCEND_DEVICE_ID
    echo "device id is ${ASCEND_DEVICE_ID}"
elif [ ${device_id} ];then
    export ASCEND_DEVICE_ID=${device_id}
    echo "device id is ${ASCEND_DEVICE_ID}"
else
    "[Error] device id must be config"
    exit 1
fi

#训练开始时间，不需要修改
start_time=$(date +%s)

#创建DeviceID输出目录，不需要修改
if [ -d ${cur_path}/output/${ASCEND_DEVICE_ID} ];then
    rm -rf ${cur_path}/output/${ASCEND_DEVICE_ID}
    mkdir -p ${cur_path}/output/$ASCEND_DEVICE_ID
else
    mkdir -p ${cur_path}/output/$ASCEND_DEVICE_ID
fi

#非平台场景时source 环境变量
check_etp_flag=`env | grep etp_running_flag`
etp_flag=`echo ${check_etp_flag#*=}`
if [ x"${etp_flag}" != x"true" ];then
     source ${cur_path}/test/env_npu.sh
fi

#执行训练脚本，以下传参不需要修改，其他需要模型审视修改
export RANK=0
let dis_bs=${batch_size}/${RANK_SIZE}
cd ${cur_path}/../
for((RANK_ID=0;RANK_ID<RANK_SIZE;RANK_ID++))
do
    export RANK=$RANK_ID

    if [ $(uname -m) = "aarch64" ]
    then
        let a=0+RANK_ID*24
        let b=23+RANK_ID*24
        taskset -c $a-$b python3.7 ./tools/train.py configs/solov2/solov2_r50_fpn_8gpu_1x.py \
            --launcher pytorch \
            --opt-level $apex \
            --gpus 8 \
            --autoscale-lr \
            --seed 0 \
            --data_root=$data_path \
            --total_epochs $epochs \
            --rt2_bin=$rt2_bin \
            --batch_size=$dis_bs \
            --fps_lag=$fps_lag \
            --interval=$interval > ${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &
    else
        python3.7 ./tools/train.py configs/solov2/solov2_r50_fpn_8gpu_1x.py \
            --launcher pytorch \
            --opt-level $apex \
            --gpus 8 \
            --autoscale-lr \
            --seed 0 \
            --data_root=$data_path \
            --total_epochs $epochs \
            --rt2_bin=$rt2_bin \
            --batch_size=$dis_bs \
            --fps_lag=$fps_lag \
            --interval=$interval > ${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &
    fi
done
wait
python3.7 tools/test_ins.py configs/solov2/solov2_r50_fpn_8gpu_1x.py  ${cur_path}/work_dirs/solov2_release_r50_fpn_8gpu_1x/epoch_${epochs}.pth --show \
      --out  results_solo.pkl --eval segm --data_root=$data_path >> ${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &
wait
#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
# FPS=`grep -a 'FPS'  $cur_path/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk -F "FPS: " '{print $NF}'|awk 'NR==1{max=$1;next}{max=max>$1?max:$1}END{print max}'`
sum_FPS=`grep 'FPS:' $cur_path/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | awk '{sum +=$NF};END{print sum}'`
num_FPS=`grep 'FPS:' $cur_path/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | wc -l`
FPS=$(awk 'BEGIN{printf "%.2f\n",'${sum_FPS}'/'${num_FPS}'}')
#打印，不需要修改
echo "Final Performance images/sec : $FPS"

#输出训练精度,需要模型审视修改
#train_accuracy=`grep  'mmcv.runner.runner:Epoch' $cur_path/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log | grep eta: | awk -F "loss: " '{print $NF}' | awk -F "," '{print $1}' | awk 'END {print}'`
train_accuracy=`grep -a 'maxDets' $cur_path/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk -F " " '{print $13}'|head -n 1`
#打印，不需要修改
echo "Final Train Accuracy : ${train_accuracy}"
echo "E2E Training Duration sec : $e2e_time"

#稳定性精度看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
if [[ $apex == "O0" ]];then
        CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'fp32'_'acc'
else
        CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'acc'
fi

##获取性能数据，不需要修改
#吞吐量
ActualFPS=${FPS}
#单迭代训练时长
TrainingTime=`awk 'BEGIN{printf "%.2f\n", '${batch_size}'*1000/'${FPS}'}'`

#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要模型审视修改
# grep Epoch: $cur_path/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|grep eta:|awk -F "loss: " '{print $NF}' | awk -F "," '{print $1}' > $cur_path/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt
grep  'mmcv.runner.runner' $cur_path/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log | grep loss | awk -F "loss: " '{print $NF}' | awk -F "," '{print $1}' > $cur_path/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt

#最后一个迭代loss值，不需要修改
ActualLoss=$(awk 'END {print}' $cur_path/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt)

#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" > $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainAccuracy = ${train_accuracy}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
