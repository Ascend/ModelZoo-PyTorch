#!/bin/bash

##################基础配置参数，需要模型审视修改##################
# 必选字段(必须在此处定义的参数): Network batch_size RANK_SIZE
#网络名称，同目录名称
Network="WideResNet101_2_for_Pytorch"
#训练epoch
train_epochs=90
#训练batch_size
batch_size=2048
#训练step
#train_steps=`expr 1281167 / ${batch_size}`
#学习率
learning_rate=0.4
# 训练使用的npu卡数
export RANK_SIZE=8
# 数据集路径,保持为空,不需要修改
data_path=""
RANK_ID_START=0
#TF2.X独有，需要模型审视修改
#export NPU_LOOP_SIZE=${train_steps}

#维测参数，precision_mode需要模型审视修改
#precision_mode="allow_mix_precision"
#维持参数，以下不需要修改
over_dump=False
data_dump_flag=False
data_dump_step="10"
profiling=False

# 帮助信息，不需要修改
if [[ $1 == --help || $1 == -h ]];then
    echo"usage:./train_performance_1P.sh <args>"
    echo " "
    echo "parameter explain:
    --precision_mode         precision mode(allow_fp32_to_fp16/force_fp16/must_keep_origin_dtype/allow_mix_precision)
    --over_dump		           if or not over detection, default is False
    --data_dump_flag		     data dump flag, default is False
    --data_dump_step		     data dump step, default is 10
    --profiling		           if or not profiling for performance debug, default is False
    --data_path		           source data of training
    -h/--help		             show help message
    "
    exit 1
fi

#参数校验，不需要修改
for para in $*
do
    if [[ $para == --workers* ]];then
        workers=`echo ${para#*=}`
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


##################创建日志输出目录，根据模型审视##################
# 模型采用非循环方式启动多卡训练，创建日志输出目录如下；采用循环方式启动多卡训练的模型，在循环中创建日志输出目录，可参考CRNN模型
# 非循环方式下8卡训练日志输出路径中的ASCEND_DEVICE_ID默认为0，只是人为指定文件夹名称， 不涉及训练业务
export ASCEND_DEVICE_ID=0
export RANK_ID=$RANK_ID_START
start_time=$(date + %s)



##################启动训练脚本##################
#训练开始时间，不需要修改
start_time=$(date +%s)
# corenum=`cat /proc/cpuinfo|grep "processor" | wc -l`
# 非平台场景时source 环境变量
check_etp_flag=`env | grep etp_running_flag`
etp_flag=`echo ${check_etp_flag#*=}`
if [ x"${etp_flag}" != x"true" ];then
    source ${test_path_dir}/env_npu.sh
fi

if [ -d ${test_path_dir}/train_8p_${start_time} ];then
    rm -rf ${test_path_dir}/train_8p_${start_time}
    mkdir -p ${test_path_dir}/train_8p_${start_time}
else
    mkdir -p ${test_path_dir}/train_8p_${start_time}
fi

python3.7 -u ./main_npu_8p.py \
    "${data_path}" \
    --addr=$(hostname -I |awk '{print $1}') \
    --lr=${learning_rate} \
    --print-freq=10 \
    --wd=0.0005 \
    --workers=128 \
    --epochs=${train_epochs} \
    --amp \
    --world-size=1 \
    --dist-backend='hccl' \
    --multiprocessing-distributed \
    --loss-scale=128.0 \
    --opt-level='O2' \
    --device='npu' \
    --rank=0 \
    --warm_up_epochs=5 \
    --save_path=${test_path_dir}/train_8p_${start_time} \
    --batch-size=${batch_size} > ${test_path_dir}/train_8p_${start_time}/train_${ASCEND_DEVICE_ID}.log 2>&1 &
wait

##################获取训练数据##################
#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

# 结果打印，不需要修改
echo "------------------ Final result ------------------"
# 输出性能FPS，需要模型审视修改
FPS=`grep -a 'FPS'  ${test_path_dir}/train_8p_${start_time}/train_${ASCEND_DEVICE_ID}.log|awk -F " " '{print $NF}'|awk 'END {print}'`
# 打印，不需要修改
echo "Final Performance images/sec : $FPS"

# 输出训练精度,需要模型审视修改
train_accuracy=`grep -a '* Acc@1' ${test_path_dir}/train_8p_${start_time}/train_${ASCEND_DEVICE_ID}.log|awk 'END {print}'|awk -F "Acc@1" '{print $NF}'|awk -F " " '{print $1}'`
# 打印，不需要修改
echo "Final Train Accuracy : ${train_accuracy}"
echo "E2E Training Duration sec : $e2e_time"

# 性能看护结果汇总
# 训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'acc'

# 获取性能数据，不需要修改
# 吞吐量
ActualFPS=${FPS}
# 单迭代训练时长
TrainingTime=`awk 'BEGIN{printf "%.2f\n", '${batch_size}'*1000/'${FPS}'}'`

# 从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
grep Epoch: ${test_path_dir}/train_8p_${start_time}/train_$ASCEND_DEVICE_ID.log|grep -v Test|awk -F "Loss" '{print $NF}' | awk -F " " '{print $1}' >> ${test_path_dir}/train_8p_${start_time}/train_${CaseName}_loss.txt

# 最后一个迭代loss值，不需要修改
ActualLoss=`awk 'END {print}' ${test_path_dir}/train_8p_${start_time}/train_${CaseName}_loss.txt`

# 关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" > ${test_path_dir}/train_8p_${start_time}/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >> ${test_path_dir}/train_8p_${start_time}/${CaseName}.log
echo "BatchSize = ${BatchSize}" >> ${test_path_dir}/train_8p_${start_time}/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> ${test_path_dir}/train_8p_${start_time}/${CaseName}.log
echo "CaseName = ${CaseName}" >> ${test_path_dir}/train_8p_${start_time}/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >> ${test_path_dir}/train_8p_${start_time}/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >> ${test_path_dir}/train_8p_${start_time}/${CaseName}.log
echo "TrainAccuracy = ${train_accuracy}" >> ${test_path_dir}/train_8p_${start_time}/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> ${test_path_dir}/train_8p_${start_time}/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> ${test_path_dir}/train_8p_${start_time}/${CaseName}.log
