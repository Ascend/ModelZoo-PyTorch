#!/bin/bash

#当前路径,不需要修改
cur_path=`pwd`

#集合通信参数,不需要修改
#保证rank table file 文件rank_table_8p.json存放在和test同级的configs目录下
export RANK_SIZE=8
batch_size=8192
#RANK_TABLE_FILE=${cur_path}/../configs/rank_table_8p.json
RANK_ID_START=0

# 数据集路径,保持为空,不需要修改
data_path=""

#设置默认日志级别,不需要修改
export ASCEND_GLOBAL_LOG_LEVEL=3

#基础参数 需要模型审视修改
#网络名称，同目录名称
Network="Shufflenetv2_ID0099_for_PyTorch"
#训练epoch
train_epochs=1
device_id_list=0,1,2,3,4,5,6,7
#TF2.X独有，不需要修改
#export NPU_LOOP_SIZE=${train_steps}

#维测参数，precision_mode需要模型审视修改
precision_mode="allow_mix_precision"
#维持参数，以下不需要修改
over_dump=False
data_dump_flag=False
data_dump_step="10"
profiling=False
autotune=False

# 帮助信息，不需要修改
if [[ $1 == --help || $1 == -h ]];then
    echo"usage:./train_full_8p.sh <args>"
    echo " "
    echo "parameter explain:
    --precision_mode           precision mode(allow_fp32_to_fp16/force_fp16/must_keep_origin_dtype/allow_mix_precision)
    --over_dump		           if or not over detection, default is False
    --data_dump_flag		   data dump flag, default is 0
    --data_dump_step		   data dump step, default is 10
    --profiling		           if or not profiling for performance debug, default is False
    --autotune                 whether to enable autotune, default is False
    --data_path		           source data of training
    -h/--help		           show help message
    "
    exit 1
fi

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
    elif [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    fi
done

#校验是否传入data_path,不需要修改
if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
    exit 1
fi

##################指定训练脚本执行路径##################
# cd到与test文件同层级目录下执行脚本，提高兼容性；test_path_dir为包含test文件夹的路径
cur_path=`pwd`
cur_path_last_dirname=${cur_path##*/}
if [ x"${cur_path_last_dirname}" == x"test" ]; then
    test_path_dir=${cur_path}
    cd ..
    cur_path=`pwd`
else
    test_path_dir=${cur_path}/test
fi

#################创建日志输出目录，不需要修改#################
ASCEND_DEVICE_ID=0
if [ -d ${test_path_dir}/output/${ASCEND_DEVICE_ID} ];then
    rm -rf ${test_path_dir}/output/${ASCEND_DEVICE_ID}
    mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID/ckpt
else
    mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID/ckpt
fi

#训练开始时间，不需要修改
start_time=$(date +%s)
# 非平台场景时source 环境变量
check_etp_flag=`env | grep etp_running_flag`
etp_flag=`echo ${check_etp_flag#*=}`
if [ x"${etp_flag}" != x"true" ];then
    source ${test_path_dir}/env_npu.sh
fi

KERNEL_NUM=$(($(nproc)/8))
for((RANK_ID=$RANK_ID_START;RANK_ID<RANK_SIZE;RANK_ID++));
do
if [ $(uname -m) = "aarch64" ]
then
    PID_START=$((KERNEL_NUM * RANK_ID))
    PID_END=$((PID_START + KERNEL_NUM - 1))
    export RANK_ID=$RANK_ID
    export DEVICE_ID=$RANK_ID
    nohup taskset -c $PID_START-$PID_END python3.7 -u ./8p_main_med.py \
        --data=$data_path \
        --addr=$(hostname -I |awk '{print $1}') \
        --seed=49  \
        --workers=$(nproc) \
        --learning-rate=4 \
        --print-freq=1 \
        --eval-freq=5 \
        --arch=shufflenet_v2_x1_0  \
        --dist-url='tcp://127.0.0.1:50000' \
        --dist-backend='hccl' \
        --multiprocessing-distributed \
        --world-size=1 \
        --batch-size=${batch_size} \
        --epochs=${train_epochs} \
        --warm_up_epochs=5 \
        --device_num=8 \
        --rank=0 \
        --amp \
        --momentum=0 \
        --device-list=${device_id_list} \
        --local_rank=${RANK_ID} \
        --num-classes=1000 \
        --benchmark 0 \
        > ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &
else
    nohup python3.7 -u ./8p_main_med.py \
        --data=$data_path \
        --addr=$(hostname -I |awk '{print $1}') \
        --seed=49  \
        --workers=$(nproc) \
        --learning-rate=4 \
        --print-freq=1 \
        --eval-freq=5 \
        --arch=shufflenet_v2_x1_0  \
        --dist-url='tcp://127.0.0.1:50000' \
        --dist-backend='hccl' \
        --multiprocessing-distributed \
        --world-size=1 \
        --batch-size=${batch_size} \
        --epochs=${train_epochs} \
        --warm_up_epochs=5 \
        --device_num=8 \
        --rank=0 \
        --amp \
        --momentum=0 \
        --device-list=${device_id_list} \
        --local_rank=${RANK_ID} \
        --num-classes=1000 \
        --benchmark 0 \
        > ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &
fi
done 
wait

#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

#稳定性精度看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}${name_bind}_bs${BatchSize}_${RANK_SIZE}'p'_'perf'

##获取性能数据

FPS=`grep "FPS@all" ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | awk 'END {print $7}'|tr -d ,`
ActualFPS=`echo "${FPS}"|bc`

# 打印，不需要修改
echo "Final Performance images/sec : $FPS"
echo "E2E Training Duration sec : $e2e_time"
TrainingTime=`echo "scale=2;${batch_size} / ${ActualFPS}"|bc`

ActualLoss=`grep "Loss" ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | awk 'END {print $12}'`

#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" > ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
