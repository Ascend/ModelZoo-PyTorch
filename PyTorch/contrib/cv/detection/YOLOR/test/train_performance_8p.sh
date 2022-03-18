#!/bin/bash
#source ./test/env_npu.sh
#设置默认日志级别,不需要修改
export ASCEND_GLOBAL_LOG_LEVEL=3
#当前路径,不需要修改
cur_path=`pwd`
export RANK_SIZE=8
RANK_SIZE=8
#RANK_TABLE_FILE=${cur_path}/../configs/rank_table_8p.json
RANK_ID_START=0

#网络名称，同目录名称
Network="YoloV5_ID0638_for_PyTorch"

# 数据集路径,保持为空,不需要修改
data_path=""
#维测参数，precision_mode需要模型审视修改
precision_mode="allow_mix_precision"
#维持参数，以下不需要修改
over_dump=False
data_dump_flag=False
data_dump_step="10"
profiling=False
autotune=False

#参数校验，不需要修改
for para in $*
do
    if [[ $para == --precision_mode* ]];then
        precision_mode=`echo ${para#*=}`
    elif [[ $para == --over_dump* ]];then
        over_dump=`echo ${para#*=}`
        over_dump_path=${cur_path}/output/overflow_dump
        mkdir -p ${over_dump_path}
    elif [[ $para == --data_dump_flag* ]];then
        data_dump_flag=`echo ${para#*=}`
        data_dump_path=${cur_path}/output/data_dump
        mkdir -p ${data_dump_path}
    elif [[ $para == --data_dump_step* ]];then
        data_dump_step=`echo ${para#*=}`
    elif [[ $para == --profiling* ]];then
        profiling=`echo ${para#*=}`
        profiling_dump_path=${cur_path}/output/profiling
        mkdir -p ${profiling_dump_path}
    elif [[ $para == --autotune* ]];then
        autotune=`echo ${para#*=}`
		export autotune=$autotune
        mv $install_path/fwkacllib/data/rl/Ascend910/custom $install_path/fwkacllib/data/rl/Ascend910/custom_bak
        mv $install_path/fwkacllib/data/tiling/Ascend910/custom $install_path/fwkacllib/data/tiling/Ascend910/custom_bak
        autotune_dump_path=${cur_path}/output/autotune_dump
        mkdir -p ${autotune_dump_path}/GA
        mkdir -p ${autotune_dump_path}/rl
        cp -rf $install_path/fwkacllib/data/tiling/Ascend910/custom ${autotune_dump_path}/GA/
        cp -rf $install_path/fwkacllib/data/rl/Ascend910/custom ${autotune_dump_path}/RL/
    elif [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    elif [[ $para == --bind_core* ]]; then
        bind_core=`echo ${para#*=}`
        name_bind="_bindcore"
    fi
done


#校验是否传入data_path,不需要修改
if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
    exit 1
fi

#autotune时，先开启autotune执行单P训练，不需要修改
if [[ $autotune == True ]]; then
    train_full_1p.sh --autotune=$autotune --data_path=$data_path
    wait
    autotune=False
	export autotune=$autotune
fi


#训练开始时间，不需要修改
start_time=$(date +%s)

#进入训练脚本目录
cd $cur_path/../
export SIll=1
for((RANK_ID=$RANK_ID_START;RANK_ID<$((SIll+RANK_ID_START));RANK_ID++));
do
    #设置环境变量，不需要修改
    echo "Device ID: $ASCEND_DEVICE_ID"
    export RANK_ID=$RANK_ID
    export DEVICE_ID=$RANK_ID
	
    #创建DeviceID输出目录，不需要修改
    if [ -d ${cur_path}/output/${ASCEND_DEVICE_ID} ];then
        rm -rf ${cur_path}/output/${ASCEND_DEVICE_ID}
        mkdir -p ${cur_path}/output/$ASCEND_DEVICE_ID/ckpt
        mkdir -p ${cur_path}/output/overflow_dump
    else
        mkdir -p ${cur_path}/output/$ASCEND_DEVICE_ID/ckpt
        mkdir -p ${cur_path}/output/overflow_dump
    fi
    over_dump_path=${cur_path}/output/overflow_dump

    python3.7 train.py \
        --data data/coco.yaml \
        --cfg cfg/yolor_p6.cfg \
        --weights '' \
        --batch-size 8 \
        --img 1280 1280 \
        --device npu \
        --npu 0 \
        --epochs 1 \
        > ${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &

done 
wait

#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))
echo "E2E Training Duration sec : $e2e_time"

#训练用例信息，不需要修改
RANK_SIZE=8
BatchSize=64
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'perf'

FPS=`grep "FPS:" $cur_path/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | awk -F "FPS:" '{print $2}'|awk 'END {print $1}' |cut -c 1-3`
ActualFPS=`echo "8 * ${FPS}"|bc`

TrainingTime=`echo "scale=2;${BatchSize} / ${ActualFPS}"|bc`

ActualLoss=`grep "totalLoss:" $cur_path/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | awk 'END {print $7}' |tr -d "totalLoss:"`

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

rm -rf $cur_path/../runs
rm -rf $data_path/labels/*.cache3